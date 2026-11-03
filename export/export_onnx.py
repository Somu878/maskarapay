"""Step 3: export BertForTokenClassification -> assets/maskara-f32.onnx.

Preferred route: optimum CLI (`optimum[-cli] export onnx --task token-classification`).
Sanctioned fallback: hand-written torch.onnx.export wrapper (exact plan recipe).
Either route is followed by a canonical post-process step:
  - graph output: exactly one, named "logits"
  - graph inputs named input_ids / attention_mask / token_type_ids, all int64
  - verified loadable by onnxruntime and runnable on a dummy 1x512 input.
"""

import os
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
MODEL_SOURCE = os.path.join(HERE, "model-source")
F32_PATH = os.path.join(ASSETS, "maskara-f32.onnx")
TMP_OPT = os.path.join(HERE, "tmp-opt")

CANON_INPUTS = ["input_ids", "attention_mask", "token_type_ids"]


# --------------------------------------------------------------------------
# Route A: optimum CLI
# --------------------------------------------------------------------------

def _find_optimum_cli(venv_bin: str) -> str | None:
    for name in ("optimum-cli", "optimum"):
        p = os.path.join(venv_bin, name)
        if os.path.exists(p):
            return p
    return None


def export_via_optimum(out_tmp: str, timeout: int = 900) -> tuple[bool, str]:
    cli = _find_optimum_cli(os.path.join(HERE, ".venv", "bin"))
    if cli is None:
        return False, "no optimum CLI binary in venv"
    cmd = [cli, "export", "onnx", "--task", "token-classification", "--model", MODEL_SOURCE]
    # optimum >= 1.20 takes the output dir/clusters as the last positional arg;
    # older syntax used --output. Cover both.
    cmd += [out_tmp]
    log_tail = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log_tail = (proc.stdout + "\n" + proc.stderr)[-2500:]
        if proc.returncode != 0:
            return False, f"exit code {proc.returncode}\n{log_tail}"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s\n{log_tail}"
    # locate the model (model.onnx expected)
    cand = os.path.join(out_tmp, "model.onnx")
    if not os.path.exists(cand):
        # search recursively as a safety net
        found = _find_onnx(out_tmp)
        if not found:
            return False, f"no model.onnx in tmp dir\n{log_tail}"
        cand = found
    return True, cand


def _find_onnx(root: str) -> str | None:
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f == "model.onnx":
                return os.path.join(dirpath, f)
    return None


# --------------------------------------------------------------------------
# Route B: sanctioned hand export
# --------------------------------------------------------------------------

def export_via_torch(out_path: str) -> tuple[bool, str]:
    import torch
    from transformers import AutoModelForTokenClassification

    try:
        model = AutoModelForTokenClassification.from_pretrained(MODEL_SOURCE)
        model.eval()

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask, token_type_ids):
                return self.m(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).logits

        wrap = Wrap(model)
        dummy_ids = torch.ones((1, 16), dtype=torch.int64)
        dummy_mask = torch.ones((1, 16), dtype=torch.int64)
        dummy_types = torch.zeros((1, 16), dtype=torch.int64)
        dynamic_axes = {}
        for nm in CANON_INPUTS:
            dynamic_axes[nm] = {0: "batch", 1: "sequence"}
        with torch.no_grad():
            torch.onnx.export(
                wrap,
                (dummy_ids, dummy_mask, dummy_types),
                out_path,
                opset_version=17,
                do_constant_folding=True,
                input_names=CANON_INPUTS,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
            )
        return True, out_path
    except Exception as e:  # noqa: BLE001 - report and let caller decide
        import traceback

        return False, traceback.format_exc()[-2500:] or str(e)


# --------------------------------------------------------------------------
# Canonical post-processing + verification
# --------------------------------------------------------------------------

def _rename_all_refs(graph, old: str, new: str):
    """Rename a graph input/output value `old` -> `new` everywhere."""
    if graph.name == old:
        graph.name = new
    for val_info in list(graph.input) + list(graph.output):
        if val_info.name == old:
            val_info.name = new
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new
    for init in graph.initializer:
        if init.name == old:
            init.name = new


def postprocess_and_verify(path: str) -> str:
    """Enforce canonical I/O names; verify shapes/dtypes; return report string."""
    model = onnx.load(path)

    # --- graph inputs ---
    infos = model.graph.input
    assert len(infos) == 3, f"expected exactly 3 graph inputs, got {len(infos)}: {[i.name for i in infos]}"
    mapping = {}
    for canonical, info in zip(CANON_INPUTS, infos):
        if info.name != canonical:
            mapping[info.name] = canonical
    for old, new in mapping.items():
        _rename_all_refs(model.graph, old, new)

    # dtypes/dims: all int64, sequence dim dynamic
    for info in model.graph.input:
        t = info.type.tensor_type
        assert t.elem_type == onnx.TensorProto.INT64, f"{info.name} is not int64"
        dims = [d.HasField("dim_param") and (d.dim_param or "") for d in t.shape.dim]
        assert len(t.shape.dim) == 2, f"{info.name} must be [B,S], got {len(t.shape.dim)} dims"

    # --- graph outputs: exactly one named logits ---
    outs = list(model.graph.output)
    assert len(outs) == 1, f"expected exactly 1 graph output, got {len(outs)}: {[o.name for o in outs]}"
    o = outs[0]
    assert o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, "logits must be float"
    assert len(o.type.tensor_type.shape.dim) == 3, "logits must be [B,S,35]"
    if o.name != "logits":
        _rename_all_refs(model.graph, o.name, "logits")

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    # re-load through onnx and save (normalizes any odd bytes)
    model2 = onnx.load(path)
    onnx.checker.check_model(model2)
    onnx.save(model2, path)

    # --- finalReload + ORT dry run 1x512 ---
    m = onnx.load(path)
    names_in = [i.name for i in m.graph.input]
    names_out = [o.name for o in m.graph.output]
    assert names_in == CANON_INPUTS, f"inputs not canonical: {names_in}"
    assert names_out == ["logits"], f"outputs not canonical: {names_out}"

    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ids = np.random.RandomState(0).randint(0, 2000, size=(1, 512)).astype(np.int64)
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones((1, 512), dtype=np.int64),
        "token_type_ids": np.zeros((1, 512), dtype=np.int64),
    }
    logits = sess.run(["logits"], feeds)[0]
    assert logits.shape == (1, 512, 35), f"bad logits shape {logits.shape}"
    assert logits.dtype == np.float32
    return (
        f"postprocess OK; inputs={names_in} (int64), output={names_out} shape {logits.shape}; "
        f"ORT 1x512 dummy run OK"
    )


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------

def build_f32(assets_dir: str | None = None, skip_optimum: bool = False) -> str:
    global ASSETS
    if assets_dir:
        ASSETS = assets_dir
        FULL = os.path.join(ASSETS, "maskara-f32.onnx")
    else:
        FULL = F32_PATH
    os.makedirs(ASSETS, exist_ok=True)

    route, detail = "none", ""
    if os.path.exists(FULL):
        try:
            rep = postprocess_and_verify(FULL)
            print(f"[export_onnx] existing model verified: {rep}")
            return FULL
        except Exception as e:  # noqa: BLE001
            print(f"[export_onnx] existing model invalid, re-exporting: {e!r}")

    if not skip_optimum:
        t0 = time.time()
        if os.path.exists(TMP_OPT):
            shutil.rmtree(TMP_OPT)
        ok, detail = export_via_optimum(TMP_OPT)
        if ok:
            src = detail
            shutil.copyfile(src, FULL)
            shutil.rmtree(TMP_OPT, ignore_errors=True)
            rep = postprocess_and_verify(FULL)
            print(f"[export_onnx] optimum route OK in {time.time()-t0:.1f}s: {rep}")
            return FULL
        print(f"[export_onnx] optimum route FAILED -> falling back to torch.onnx.export:\n{detail}")
    else:
        print("[export_onnx] optimum skipped by flag")

    t1 = time.time()
    ok, detail = export_via_torch(FULL)
    if not ok:
        raise RuntimeError(f"torch export failed:\n{detail}")
    rep = postprocess_and_verify(FULL)
    print(f"[export_onnx] torch fallback OK in {time.time()-t1:.1f}s: {rep}")
    return FULL


if __name__ == "__main__":
    skip = "--skip-optimum" in sys.argv
    build_f32(skip_optimum=skip)
