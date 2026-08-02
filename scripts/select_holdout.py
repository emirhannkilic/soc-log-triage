"""
v3 plan section 3.3: selects 30 candidate emails (15 phishing, 15 legitimate)
for the hold-out set, from the already-parsed+anonymized facts
(data/processed/phishing_facts.jsonl, gmail_facts.jsonl).

Deliberately picks a MIX of easy and hard cases per class rather than random
sampling, so the hold-out actually exercises the rule engine instead of
letting it get away with a single strong signal (e.g. SPF alone):

Phishing (15): stratified across spf_result buckets, weighted toward
"hard" cases (spf=pass, i.e. sent from attacker-controlled infra that
passes its own SPF check — the plan's section 0.1 point that SPF alone
isn't a reliable phishing/legit split) without excluding the "easy"
spf=fail majority.

Legitimate (15): stratified across whether the sender domain is a
recognizable brand vs. a personal contact, since the rule engine's
display_name_brand_mismatch check needs both kinds represented.

holdout-fix-tasks.md T6: the raw phishing_pot pool mixes real credential
phishing with plain commercial spam (language courses, adult dating,
detox products, horoscopes, casino/gambling ads, health-supplement
clickbait, cold B2B sales pitches, prize/survey scams — no brand
impersonation, no credential request, no phishing intent). Random
stratified picking surfaced 4 such spam records in the original
30-candidate set, which taught the wrong lesson (e.g. "German marketing =
phishing, Turkish marketing = legitimate" instead of any real phishing
signal).

Fixing this turned out to need more than "exclude scripts/
audit_spam_vs_phishing.py's likely_spam records": hand-checking each
re-drawn candidate set repeatedly surfaced spam the audit's keyword
heuristic missed — the same few spam campaigns recur under many different
sample-N.eml IDs, and the audit's fallback for "no spam keyword AND no
phishing signal matched" is to assume phishing, which let weak/ambiguous
junk through untouched. So the pool is filtered several ways at once (see
the constants below): audit likely_spam, a known spam-infrastructure
sender the audit misses entirely (stayfriends.de), known recurring spam
templates matched by content rather than sample ID, a body-usability
check (empty/garbled bodies can't be hand-labeled either way), and
finally requiring an actual POSITIVE phishing signal in the audit's
reasoning rather than merely "wasn't flagged as spam". Commercial spam
can still enter the holdout, but only deliberately, via
FORCE_INCLUDE_PHISHING_PATHS. Every phishing candidate also carries
is_spam_not_phishing in its holdout metadata (T6 item 1), so a reviewer
who disagrees with all of the above during hand-labeling can see and
correct it rather than have the disagreement silently lost.

Output: data/holdout/candidates.jsonl — each line has the anonymized facts
PLUS the original relative .eml path, so a human reviewer can open the raw
file to read full context while writing the expected JSON verdict.
"""
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROJECT_ROOT / "data" / "holdout" / "candidates.jsonl"
SPAM_AUDIT_PATH = PROCESSED_DIR / "phishing_facts_spam_audit.jsonl"

PHISHING_COUNT = 15
LEGITIMATE_COUNT = 15
SEED = 7

# holdout-fix-tasks.md T6 item 2: 4 commercial-spam records the original
# random selection surfaced (German language-course ad, cooking-course ad,
# adult-dating spam, detox-product ad — all from the same stayfriends.de
# spam-infrastructure sender), replaced with real credential-phishing
# examples: brand impersonation (Microsoft signin alert, homoglyph
# "Someone tried to Iog in" Facebook phish) and credential/account asks
# tied to a specific service (USDT wallet withdrawal, Netflix payment
# update) — diverse from the phishing already in the pool (avoids
# duplicating the existing Microsoft/Banco do Brasil/mailbox-relogin
# examples in flavor where possible).
FORCE_INCLUDE_PHISHING_PATHS = [
    "data/phishing_pot/email/sample-1299.eml",  # Microsoft signin alert
    "data/phishing_pot/email/sample-789.eml",   # homoglyph Facebook credential phish
    "data/phishing_pot/email/sample-1212.eml",  # USDT/crypto wallet phishing
    "data/phishing_pot/email/sample-3149.eml",  # Netflix payment-update phishing
]

# Hand-verified spam (read in full, not just heuristic output) that MUST
# never re-enter the holdout, overriding whatever
# scripts/audit_spam_vs_phishing.py says. This matters concretely: the
# audit's own heuristic misses all 4 of these — its "fake urgency + action
# link" rule fires on the same boilerplate-template garbage
# (24 hours / expire / click here fragments unrelated to the actual spam
# content) that made credential_request produce false positives on this
# exact stayfriends.de sender before T3. The audit is a useful coarse
# filter for the OTHER ~1490 records, but it is not trusted blindly for
# records a human has actually read.
HAND_VERIFIED_SPAM_PATHS = {
    "data/phishing_pot/email/sample-2428.eml",  # German language-course spam
    "data/phishing_pot/email/sample-2002.eml",  # German cooking-course spam
    "data/phishing_pot/email/sample-2023.eml",  # German adult-dating spam
    "data/phishing_pot/email/sample-3436.eml",  # German detox-product spam
    "data/phishing_pot/email/sample-7792.eml",  # "Heated Vest" product spam, no brand/credential ask
    "data/phishing_pot/email/sample-2551.eml",  # Coco Chanel prize-survey spam, no real brand actor
}

# The first 4 above (from stayfriends.de) turned out to be one instance of
# a bigger problem: stayfriends.de is a spam-infrastructure sender running
# a large, repetitive campaign (69/1500 records in the phishing_pot
# sample) — dating/hookup spam, marketing giveaways, horoscopes — that the
# audit heuristic systematically misses (its "fake urgency + action link"
# rule fires on boilerplate template fragments the campaign reuses across
# otherwise unrelated ad content, e.g. "24 hours"/"expire"/"click here"
# strings stitched in from an unrelated template). Re-drawing the holdout
# pool after excluding just the first 4 paths surfaced MORE stayfriends.de
# spam (duplicate "heiße ukrainische Singles" ads, "VollNaße6Pussy" adult
# content) sitting right behind them. Rather than hand-verify all 69,
# the entire sender is excluded from the holdout pool — it isn't a loss
# of phishing diversity, since none of its records were real phishing.
SPAM_INFRASTRUCTURE_DOMAINS = {"stayfriends.de"}

# Hand-checking individual picks kept surfacing the SAME spam campaigns
# under different sample-N.eml IDs (e.g. the "Consumer Winner / Vous avez
# été choisi pour recevoir un cadeau" prize-survey template, the
# "OpenClaw"/cold-sales-pitch template) — the phishing_pot dataset contains
# many near-duplicate copies of a small number of spam templates, so a
# one-off path blocklist keeps losing to re-draws that pick a different
# copy of the same template. Matching the template text itself, rather
# than specific sample IDs, is what actually holds across re-draws.
_SPAM_TEMPLATE_RE = re.compile(
    r"vous avez ét[ée] choisi|f[ée]licitations|"        # prize-survey template
    r"pay only if it runs more reliably|"                # cold B2B sales-pitch template
    r"free spins|gambling is|new players|"                # online-casino ad template
    r"afspreken via whatsapp|dit werkt beter dan tinder|"  # dating-app ad template
    r"breakthrough.{0,20}ritual|you won.t believe",         # health/supplement clickbait ad
    re.IGNORECASE,
)

# scripts/audit_spam_vs_phishing.py's fallback for records matching
# neither a spam-topic keyword NOR a phishing signal is to default to
# "phishing" (likely_spam=False) rather than drop them — reasonable for
# the T6-item-4 corpus-wide ratio estimate (undercounting spam there is
# the safer error), but wrong for holdout selection: 411/865 of its
# "not spam" records carry exactly this reason string, and hand-checking
# repeatedly found real spam/junk hiding in that bucket (a near-empty
# MercadoPago loan-marketing fragment with an unsubscribe link, a German
# weight-loss-supplement ad) that happened not to match any spam keyword.
# The holdout pool requires an actual POSITIVE phishing signal in the
# audit's reasoning — brand impersonation, a credential/account-security
# ask, fake urgency + action link, or 419/advance-fee fraud — not just
# "wasn't flagged as spam".
_NO_SIGNAL_REASON = (
    "no strong spam-topic match but also no confirmed brand/credential/"
    "urgency phishing signal"
)

# Records with unusably short/garbled body_text (empty after strip, or raw
# undecoded base64 leaking through because get_body() picked the wrong
# MIME part) can't be hand-labeled at all regardless of spam/phishing
# status — a human reviewer needs readable content to write a verdict.
# This is a parser data-quality filter, not a T6 spam/phishing judgment.
MIN_USABLE_BODY_CHARS = 30
# Undecoded base64 leaking through as body_text has near-zero whitespace —
# a single "word" can run for the length of the whole snippet (found via
# sample-8233.eml: a 150+-char unbroken token). Real prose in any language
# doesn't produce tokens this long.
MAX_PLAUSIBLE_WORD_CHARS = 60


def _has_usable_body(body_text: str) -> bool:
    text = (body_text or "").strip()
    if len(text) < MIN_USABLE_BODY_CHARS:
        return False
    words = text.split()
    if words and max(len(w) for w in words) > MAX_PLAUSIBLE_WORD_CHARS:
        return False
    return True


def load_facts_with_path(facts_file: Path, sample_file: Path) -> list[dict]:
    facts = [json.loads(line) for line in open(facts_file) if line.strip()]
    paths = [json.loads(line)["path"] for line in open(sample_file) if line.strip()]
    # facts and sample_file lines were written in the same order by
    # parse_and_anonymize.py, so zipping by position is safe here.
    for f, p in zip(facts, paths):
        f["_eml_path"] = p
    return facts


def attach_spam_audit(records: list[dict]) -> None:
    """Merges is_spam_not_phishing (+ the audit's reasoning) into each
    phishing record's holdout metadata, in place. Best-effort: if the audit
    hasn't been run, every record defaults to is_spam_not_phishing=None
    (unknown) rather than silently claiming "not spam" — see T6 item 1,
    this must never be lost by defaulting to False."""
    if not SPAM_AUDIT_PATH.exists():
        for r in records:
            r["is_spam_not_phishing"] = None
            r["spam_reason"] = None
        return
    audit = [json.loads(line) for line in open(SPAM_AUDIT_PATH) if line.strip()]
    audit_by_path = {}
    paths = [json.loads(line)["path"]
             for line in open(PROCESSED_DIR / "phishing_sample.jsonl") if line.strip()]
    for a, p in zip(audit, paths):
        audit_by_path[p] = a
    for r in records:
        if r["_eml_path"] in HAND_VERIFIED_SPAM_PATHS:
            r["is_spam_not_phishing"] = True
            r["spam_reason"] = "hand-verified commercial spam (holdout-fix-tasks.md T6)"
            continue
        a = audit_by_path.get(r["_eml_path"])
        r["is_spam_not_phishing"] = a["likely_spam"] if a else None
        r["spam_reason"] = a["spam_reason"] if a else None


def stratified_pick(records: list[dict], key_fn, count: int, rng: random.Random) -> list[dict]:
    buckets: dict = {}
    for r in records:
        buckets.setdefault(key_fn(r), []).append(r)

    picked = []
    keys = list(buckets.keys())
    rng.shuffle(keys)
    i = 0
    while len(picked) < count and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            picked.append(buckets[k].pop(rng.randrange(len(buckets[k]))))
        i += 1
    return picked


def main() -> None:
    rng = random.Random(SEED)

    phishing = load_facts_with_path(
        PROCESSED_DIR / "phishing_facts.jsonl",
        PROCESSED_DIR / "phishing_sample.jsonl",
    )
    legitimate = load_facts_with_path(
        PROCESSED_DIR / "gmail_facts.jsonl",
        PROCESSED_DIR / "gmail_sample.jsonl",
    )
    attach_spam_audit(phishing)

    # T6: force-include the hand-verified replacement phishing examples
    # first, then draw the rest of the pool from records that are (a) not
    # flagged likely_spam by the audit, (b) not from a known
    # spam-infrastructure sender the audit misses, (c) not a known
    # recurring spam template under a different sample ID, (d) have a
    # usable (non-empty, non-garbled) body, and (e) have an actual
    # POSITIVE phishing signal in the audit's reasoning rather than just
    # "wasn't flagged as spam" (see _NO_SIGNAL_REASON) — so commercial
    # spam and unlabelable junk can only enter the holdout via
    # FORCE_INCLUDE_PHISHING_PATHS (a deliberate, reviewable decision),
    # never by accident.
    forced = [r for r in phishing if r["_eml_path"] in FORCE_INCLUDE_PHISHING_PATHS]
    forced_paths = {r["_eml_path"] for r in forced}
    remaining_pool = [
        r for r in phishing
        if r["_eml_path"] not in forced_paths
        and not r["is_spam_not_phishing"]
        and r["from_domain"] not in SPAM_INFRASTRUCTURE_DOMAINS
        and not _SPAM_TEMPLATE_RE.search(r.get("body_text") or "")
        and _has_usable_body(r.get("body_text"))
        and not (r.get("spam_reason") or "").startswith(_NO_SIGNAL_REASON)
    ]

    # spf_result buckets, "pass" is the hard case for phishing (attacker
    # infra with valid SPF for its own domain).
    phishing_picked = forced + stratified_pick(
        remaining_pool, lambda r: r["spf_result"], PHISHING_COUNT - len(forced), rng
    )

    # brand-like display name vs. not, so both rule-engine paths get
    # exercised on the legitimate side too.
    def is_brand_like(r: dict) -> bool:
        name = (r.get("display_name") or "").lower()
        # crude heuristic just for stratifying the holdout selection, not
        # a parser fact — a capitalized business-looking name vs. a
        # personal "First Last" pattern is good enough here.
        return bool(name) and not (len(name.split()) == 2 and name.istitle())

    legitimate_picked = stratified_pick(
        legitimate, is_brand_like, LEGITIMATE_COUNT, rng
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in phishing_picked:
            f.write(json.dumps({"source_label": "phishing", **r}, ensure_ascii=False) + "\n")
        for r in legitimate_picked:
            f.write(json.dumps({"source_label": "legitimate", **r}, ensure_ascii=False) + "\n")

    print(f"Selected {len(phishing_picked)} phishing + {len(legitimate_picked)} legitimate "
          f"candidates -> {OUT_PATH}")
    print("spf_result buckets in phishing selection:",
          sorted({r["spf_result"] for r in phishing_picked}, key=str))

    spam_flagged = sum(1 for r in phishing_picked if r["is_spam_not_phishing"])
    print(f"is_spam_not_phishing=True in phishing selection: {spam_flagged} "
          f"(T6 threshold: <=2; forced-include examples are always False by construction)")

    if SPAM_AUDIT_PATH.exists():
        pool_spam = sum(1 for r in phishing if r["is_spam_not_phishing"])
        print(f"T6 item 4 — spam ratio across full {len(phishing)}-record phishing_pot "
              f"sample: {pool_spam}/{len(phishing)} ({pool_spam / len(phishing):.1%}) "
              f"flagged likely_spam by scripts/audit_spam_vs_phishing.py "
              f"(heuristic estimate, not hand-verified ground truth)")


if __name__ == "__main__":
    main()
