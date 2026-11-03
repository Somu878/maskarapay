"""Synthetic, deterministic golden case generator for MaskaraPay.

ALL DATA IS FAKE / SYNTHETIC. No real personal data is ever used.
Deterministic: no RNG except random.Random(seed) for fixed text pattern shuffles.
"""

from __future__ import annotations

import random

# --------------------------------------------------------------------------
# helper: token-aware filler to hit exact content-token counts
# --------------------------------------------------------------------------

FILLER_WORDS = [
    "data", "team", "road", "news", "kabhi", "life", "code", "mango",
    "market", "photo", "story", "email", "video", "party", "coffee",
    "remote", "garden", "laptop", "ticket", "banner", "market",
]


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True)) - 2


def pad_to_exact_tokens(text: str, target: int, tokenizer) -> str:
    """Append single-token filler words until content count == target exactly."""
    assert count_tokens(tokenizer, text) <= target, f"base already longer than {target}"
    out = text
    i = 0
    n = count_tokens(tokenizer, out)
    while n < target:
        w = FILLER_WORDS[i % len(FILLER_WORDS)]
        # candidate words must each be exactly 1 WordPiece token
        piece = tokenizer.encode(w, add_special_tokens=False)
        if len(piece.ids) == 1:
            candidate = out.rstrip() + " " + w
            if count_tokens(tokenizer, candidate) <= target:
                out = candidate
        i += 1
        n = count_tokens(tokenizer, out)
        if n == target:
            break
        if i > 6000:
            raise RuntimeError("could not pad to exact tokens")
    assert count_tokens(tokenizer, out) == target
    return out


def pad_then_entity(intro: str, entity: str, tail: str, target_start_idx: int, tokenizer) -> str:
    """intro + filler-pad + entity + tail where the entity's first token lands
    exactly at content-token index `target_start_idx` (spec: word can be split
    by the window boundary only if we place it around 510)."""
    # token index of entity word == content token count of (intro + pad)
    n0 = count_tokens(tokenizer, intro)
    pad_needed = target_start_idx - n0
    assert pad_needed >= 0, "intro too long for this placement"
    txt = intro if intro.endswith(" ") else intro + " "
    txt = pad_to_exact_tokens(txt, n0 + pad_needed, tokenizer).rstrip() + " " + entity
    assert count_tokens(tokenizer, txt) >= target_start_idx  # sanity
    # verify placement
    import tokenizers as tk  # noqa: F401

    enc = tokenizer.encode(txt, add_special_tokens=True)
    # find first content token whose offset start == position of entity start
    ent_char_start = txt.rfind(entity)
    idxs = [i for i in range(1, len(enc.ids) - 1) if enc.offsets[i][0] == ent_char_start]
    assert idxs and idxs[0] - 1 == target_start_idx, f"placement failed: got {idxs[:1]} want {target_start_idx}"
    return txt


# --------------------------------------------------------------------------
# case text pools (all fake data)
# --------------------------------------------------------------------------

EN_SENTENCES = [
    # names
    "My colleague Emily Carter will lead the review meeting tomorrow at noon.",
    "Contact Sarah Patel before Friday if the shipment has not arrived yet.",
    "James O'Conner and his brother Daniel joined the workshop in march.",
    "The manager Priya Nair approved the budget raise last week quietly.",
    "You can reach Marco Bianchi through the Lisbon office any weekday.",
    "Dr. Hannah Kim presented the quarterly results in Singapore on stage.",
    # emails
    "Send the signed form to alex.morgan@example.com before Monday sharp.",
    "The ticket went to support+urgent@helpdesk.io and nobody replied there.",
    "Reply both to me at john.doe.mail@corp-example.org and to the auditors.",
    "Use the alias bookings41@travel-net.co.uk instead of the shared inbox.",
    "Password reset link was emailed to fatima.zahra@example.org promptly.",
    "The newsletter list contains ravi_1990@quickmail.in and 4 others.",
    # phones
    "Call the clinic desk at +1 415 555 0123 during business hours only.",
    "Emergency contact number is (212) 555-0199, ask for reception desk.",
    "Text me at 440-555-8821 when the courier reaches the front gate.",
    "Their hotline +6 12 3456 7890 rings unanswered most of the afternoon.",
    "Order updates go to 07700 900123, the temporary travel number.",
    # companies
    "He works at GlobeSoft Pvt Ltd in the payments integration department.",
    "The vendor invoice came from Acme Trading Company Limited yesterday.",
    "Northwind Analyticals Inc. published the compliance summary in May.",
    "She joined Blue Orchid Media LLP after the merger was completed.",
    "Cipher Security Group won the audit tender for the fiscal year.",
    "Our supplier Delta Fresh Foods Private Limited raised prices again.",
    # addresses
    "Ship it to 221B Baker Street, London NW1 6XE, United Kingdom please.",
    "Her office is at 12 Queen Victoria Road, Sydney NSW 2050, floor three.",
    "The warehouse address is 458 Industrial Park Ave, Unit 7B, Austin TX.",
    "Registered office: 742 Evergreen Terrace, Springfield, OR 97477, USA.",
    "Mail branch copies to 18 Yorkville Ave, Toronto, ON M4W 1L6, Canada.",
    # dates
    "The contract starts on 2024-11-03 and expires after twelve months.",
    "Remember the audit is scheduled for 31 December 2023 without fail.",
    "He was born on 04/07/1990 according to the hospital record copy.",
    "Renewal reminder lands on 15-08-2027 in the shared team calendar.",
    # IP
    "The webhook endpoint listens on 192.168.0.42 through the office LAN.",
    "Blocked traffic from 10.0.4.19 and 203.0.113.77 for security reasons.",
    "Log shows a bounce from 172.16.254.1 in the payments shard today.",
    # ssn
    "US form asked to verify SSN 545-88-1201 against the archive copies.",
    "Application field filled as SSN 890123456 with two references.",
    # credit card
    "Backup card on file is 4539 8877 1122 3344, Visa network, expiring soon.",
    "Charges on 5555444444446666 were reversed by the bank on Monday.",
    "The virtual card 4242-4242-4242-4242 is limited to gasoline only.",
    # api keys
    "Rotate the API key sk_demo_maskara_synthetic_key_0001 before rollout.",
    "Env var stores the old token github_demo_token_not_a_secret_0001.",
    "The webhook secret is webhook_demo_secret_not_a_key_0001 configured.",
    # usernames
    "Handle is @funny_panda_42 on social, mirrored to other networks.",
    "Login as user devkabir and the vault will expose the scripts page.",
    "Their profile links to github.com/arjunkanna from the footer badge.",
    # passport (global format)
    "For the visa file, passport number L8822041 was mentioned in e-mail.",
    "Her UK passport XN4413002 expired before the internship started.",
    # vehicle reg (global)
    "Cab number plate CA 4NP 828 was parked near the loading dock area.",
]

IN_SENTENCES = [
    "Aadhaar card number 2314 5678 9012 must not be shared on WhatsApp groups.",
    "During the Camp registration they insisted on entering aadhaar 482594763157 digits only.",
    "My aadhaaar front copy had 997711442266 printed under the QR code area.",
    "PAN card ABCDE1234F is linked to the quarterly tax portal login finally.",
    "For the brokerage KYC, PAN like ABCDE1234F must be uppercase letters only.",
    "Bank asked to update PAN BXTPS4567K for the senior savings account.",
    "Pay the tea vendor on upi ravi.kumar@okhdfcbank before the weekend.",
    "Her handle name@ybl accepts payments round the clock from any bank app.",
    "Send my share to yourname@paytm and mark it as chai party dues.",
    "Indian passport number M1234567 was printed on the fresh teal booklet.",
    "Passport like M1234567 must have one letter then seven numeric digits.",
    "Vehicle MH12DE1433 got the service reminder by SMS this morning.",
    "Cab MH12DE1433 is assigned to the airport route tomorrow at noon.",
    "Call me back on +91 9876543210 regarding the flat booking refund.",
    "Phone variants like 022-26261111 or +919873451230 reach the BPO desk.",
]

HINGLISH_SENTENCES = [
    "mera naam Ravi hai, call karo aaj raat ko.",
    "aaj Chai peene chalo, weather acha hai.",
    "Mummy ne kaha hai ki dukan waale se udhaar mat lena, Ravi.",
    "Ravi bhai ne kaha tha ki account me paisa aaj aa jayega bhai.",
    "Weather ka bahut kharab scene hai, Kal na jaao office, Ravi.",
    "Bhai ne bola Ravi se: khana kahake jaldi so jao aaj.",
    "Mummy said Ravi will handle the electricity bill payment online, chalo.",
    "Aaj raat chai peene chalo, sab yaar saath me, weather bhi mast hai.",
]

STRESS_SENTENCES = [
    "Ravi Sharma, email ravi.sharma@example.com, phone +91 9876543210, seat 12A.",
    "Guest list: Emily Carter (ID 545-88-1201), card 4539 8877 1122 3344, IP 192.168.0.42.",
    "Two users shared the console: devkabir and ravi_1990@quickmail.in logged in twice.",
    "Shipment from Acme Trading Company Limited reached 221B Baker Street, London NW1 6XE.",
    "On 2024-11-03, ticket owner @funny_panda_42 rotated the API key sk_demo_maskara_synthetic_key_0001.",
    "Update PAN card ABCDE1234F and Aadhaar 2314 5678 9012 in the same portal visit please.",
    "Ravi then emailed support+urgent@helpdesk.io about chargeback on 4242-4242-4242-4242.",
    "Pipeline owner Priya Nair and backup github.com/arjunkanna watched the 10.0.4.19 audit logs.",
]

OFFSET_ROBUSTNESS = [
    "Indic नमस्ते and emoji 🎉🪔 in one sentence, phone +91 9876543210 at the end.",
    "Flag 🇮🇳 +Priya Nair wrote मुझे mail करो ravi_1990@quickmail.in पे।",
    "User typed “curly quotes” around 4539 8877 1122 3344 and left — strange.",
    "Accents test: Café naïve télé bij café near 10.0.4.19 logs, only numbers matter.",
    "👨‍👩‍👧‍👦 family emoji followed by employee Emily Carter at the auditorium.",
    "Devanagari numerals mixed: बिल ₹१२३ vs litter 022-26261111 then aadhaar 2314 5678 9012.",
    "Mixed script subject: क्या हाल the ravi.kumar@okhdfcbank sent a mail on 2024-11-03.",
    "Curly apostrophe in name: D’Angelo’s Bills, phone 07700 900123 to verify it.",
    "ZWJ sequence 👩‍💻 plus PAN card ABCDE1234F screenshot captured ravi_1990@quickmail.in.",
    "Emoji mid-entity?: Emoji 😂+ James O'Conner 🚀 reached event email alex.morgan@example.com.",
    "Long dash and ellipsis: “weird — sentence…” with IP 172.16.254.1 in the footer note.",
    "Devanagari + latin: मेरा naam मोहन है, contact +919873451230 करो जल्दी।",
]

WEIRD_WHITESPACE = [
    "",
    "hello",
    "hello world foo",
    "!!! ??? ... ,,,",
    "double  spaces\tand\ta tab on 4539 8877 1122 3344 digits.",
    "line breaks\r\nin the\rcontent\nof Emily Carter address.",
    "\u00a0Non-breaking space\u00a0between\u00a0PAN card ABCDE1234F words.",
    "    leading spaces then ravi.sharma@example.com then trailing spaces.    ",
    "nbsp\u2003and\u2000various\u3000space\u2009kinds\u205fhere.",
    "zero\u200bwidth\u200bchars\u200bbefore\u200bdevkabir\u200bname",
    "\t\n\r   \u00a0 mixed whitespace only then +91 9876543210 break   \n\t",
    "emoji in whitespace soup: 🙏 –\tEmily Carter\t🚗 – 221B Baker Street, London NW1 6XE.",
]

# long-case building blocks (entities embedded on purpose; all fake)
LONG_PATTERNS = [
    "Fruit vendor Ravi Sharma emailed ravi.sharma@example.com about invoice #4521 and asked to call back at +91 9876543210 before the market closed at 06:00 PM.",
    "The account card backup list had 4539 8877 1122 3344 configured by Priya Nair from the payments pod at GlobeSoft Pvt Ltd.",
    "Invoice #4521 was merged into journal entry for PAN card ABCDE1234F with aadhaar 2314 5678 9012 verified by the back office team.",
    "Emily Carter posted at 221B Baker Street, London NW1 6XE that her card 4242-4242-4242-4242 was charged on 2024-11-03 around 06:00 PM.",
    "devkabir rotated the API key sk_demo_maskara_synthetic_key_0001 and pinned host 192.168.0.42 in the dashboard @funny_panda_42 replied ok.",
    "Tea break notes: call Priya Nair at 440-555-8821, then email james.oconner@corp-example.org about the NYSE plate CAB 4NP 828 audit.",
]
LONG_REPEAT_FILLER = [
    "Archive note: the market list was updated quietly and nobody outside the pod noticed the change.",
    "Same pattern repeats on the dashboards for the next quarter with stricter alerts attached.",
    "Both vendors agreed to keep the shared mailbox clean and forward only the relevant mail.",
]


def build_long_case(pattern_idx: int, repeats: int, seed: int) -> str:
    rng = random.Random(seed)
    rng.shuffle(LONG_REPEAT_FILLER)  # deterministic given seed
    parts = []
    for k in range(repeats):
        parts.append(LONG_PATTERNS[(pattern_idx + k) % len(LONG_PATTERNS)])
        parts.append(LONG_REPEAT_FILLER[k % len(LONG_REPEAT_FILLER)])
    return " ".join(parts)


# --------------------------------------------------------------------------
# public: ordered deterministic case list
# --------------------------------------------------------------------------

def all_cases(tokenizer) -> list[dict]:
    cases: list[dict] = []

    def add(cid: str, text: str):
        cases.append({"id": cid, "text": text})

    # --- trivial / length edges ---
    add("trivial_empty", "")
    add("trivial_single_token", "hello")
    add("trivial_three_words", "hello world foo")
    add("trivial_punct_only", "!!! ??? ... ,,,")
    add("trivial_digit", "42")
    add("trivial_uni_word", "नमस्ते")

    # --- weird whitespace ---
    for i, t in enumerate(WEIRD_WHITESPACE, 1):
        add(f"whitespace_{i:02d}", t)

    # --- offsets robustness (mixed scripts/emoji/quotes) ---
    for i, t in enumerate(OFFSET_ROBUSTNESS, 1):
        add(f"offsets_robust_{i:02d}", t)

    # --- entity zoo (english) ---
    groups = {
        "en_name": EN_SENTENCES[0:7],
        "en_email": EN_SENTENCES[6:12],
        "en_phone": EN_SENTENCES[12:17],
        "en_company": EN_SENTENCES[17:23],
        "en_address": EN_SENTENCES[23:28],
        "en_date": EN_SENTENCES[28:32],
        "en_ip": EN_SENTENCES[32:35],
        "en_ssn": EN_SENTENCES[35:37],
        "en_cc": EN_SENTENCES[37:40],
        "en_apikey": EN_SENTENCES[40:43],
        "en_username": EN_SENTENCES[43:46],
        "en_passport": EN_SENTENCES[46:48],
        "en_vehicle_global": EN_SENTENCES[48:49],
    }
    for g, lst in groups.items():
        for i, t in enumerate(lst, 1):
            add(f"{g}_{i:02d}", t)

    # --- indian context ---
    in_groups = {
        "in_aadhaar": IN_SENTENCES[0:3],
        "in_pan": IN_SENTENCES[3:6],
        "in_upi": IN_SENTENCES[6:9],
        "in_passport": IN_SENTENCES[9:11],
        "in_vehicle": IN_SENTENCES[11:13],
        "in_phone": IN_SENTENCES[13:15],
    }
    for g, lst in in_groups.items():
        for i, t in enumerate(lst, 1):
            add(f"{g}_{i:02d}", t)

    # --- hinglish ---
    for i, t in enumerate(HINGLISH_SENTENCES, 1):
        add(f"hinglish_{i:02d}", t)

    # --- multi-entity stress / BIO edges ---
    for i, t in enumerate(STRESS_SENTENCES, 1):
        add(f"stress_{i:02d}", t)

    # --- exact token-count boundaries ---
    tok = tokenizer
    # 510 content tokens -> single window with 512 total (the model max edge)
    base = STRESS_SENTENCES[0]
    t510 = pad_to_exact_tokens(base, 510, tok)
    add("boundary_510_single_window", t510)
    # 511 -> two windows, overlap 256..510
    t511 = pad_to_exact_tokens(base, 511, tok)
    add("boundary_511_two_windows", t511)
    # 512 -> two windows
    t512 = pad_to_exact_tokens(base, 512, tok)
    add("boundary_512_two_windows", t512)
    # a word/car-number placed to straddle the 510 window edge:
    straddle = pad_then_entity(
        "Order audit for store #77. ",
        "4539 8877 1122 3344",
        " was charged twice.",
        505,
        tok,
    )
    assert count_tokens(tok, straddle) == 511, count_tokens(tok, straddle)
    add("boundary_straddle_entity", straddle)
    # 257 (stride irrelevant; sanity small two-window plan)
    t257 = pad_to_exact_tokens("Contact Marco Bianchi about the rebuilt dashboard tonight. ", 257, tok)
    add("boundary_257_single_window", t257)
    # 258 -> starts stride window count check (still single window)
    t258 = pad_to_exact_tokens("Contact Marco Bianchi about the rebuilt dashboard tonight. ", 258, tok)
    add("boundary_258_single_window", t258)

    # --- long cases (>510 content tokens) ---
    long_specs = [
        ("long_0001", 0, 20),   # ~ 20 patterns ≈ 500-600 tokens (validated below)
        ("long_0002", 1, 22),
        ("long_0003", 2, 26),
        ("long_0004", 3, 30),   # pattern with address + card
        ("long_0005", 4, 24),
        ("long_0006", 5, 28),
        ("long_0007", 0, 16),   # ~ 400-500; may be < 600? validated by assert
        ("long_0008", 2, 20),
        ("long_0009", 4, 34),
    ]
    for cid, pidx, reps in long_specs:
        add(cid, build_long_case(pidx, reps, seed=42 + pidx * 7 + reps))

    # validation of invariants on the built cases
    n = len(cases)
   _long = []
    for c in cases:
        nt = count_tokens(tok, c["text"])
        c["_n_tokens"] = nt
        c["_n_windows"] = len(
            [(0, 0)] if nt <= 510 else [(s, min(s + 510, nt)) for s in range(0, nt, 256)]
        )
        if nt > 510:
            _long.append(c)

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert n >= 120, f"only {n} cases, need >= 120"
    assert _long[-1]["_n_tokens"] <= 2000, f"longest case too long: {_long[-1]['_n_tokens']}"

    # long-range requirements: >= 8 >510-token cases in the 600..2000 range
    in_range = [c for c in _long if 600 <= c["_n_tokens"] <= 2000]
    assert len(in_range) >= 8, f"only {len(in_range)} cases in 600..2000 range: " + ", ".join(
        f"{c['id']}={c['_n_tokens']}" for c in _long
    )
    return cases
