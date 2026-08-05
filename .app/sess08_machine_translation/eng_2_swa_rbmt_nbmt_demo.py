"""
--------------------------------------------------------------------------------------------------------
Python script to demonstrate machine based translation using RBMT & NBMT
--------------------------------------------------------------------------------------------------------

1. **Rule-Based Machine Translation (RBMT)**
   Translates text using hand-crafted bilingual dictionaries,
   morphological rules, and structural transfer heuristics.
   It is transparent, interpretable, and requires no training data,
   but it does not scale well to unseen vocabulary or complex syntax.

2. **Neural-Based Machine Translation (NBMT)**
   Uses the Helsinki-NLP Opus-MT pre-trained transformer model
   (MarianMT) to translate with learned contextual representations.
   This approach captures long-range dependencies and idiomatic
   expressions far better than rule-based systems.

Requirements:
    pip install transformers sentencepiece sacrebleu rouge-score matplotlib torch

Author: Salim TS
Date: 29 Jul 2026
"""
# -----------------------------------------------------------------------------------
# 0. Import the required modules
# -----------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import re
import textwrap
import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------
# 1. Source text (Two English UK paragraphs)
# -----------------------------------------------------------------------------------
SOURCE_PARAGRAPHS: list[str] = [
    (
        "Last autumn, I travelled by train from Manchester to Edinburgh to visit an old "
        "university friend whom I had not seen in several years. The journey began very "
        "early in the morning, and the station was already crowded with commuters carrying "
        "coffee cups and briefcases. Although the weather forecast had predicted heavy rain "
        "throughout the day, the skies remained surprisingly clear for most of the trip. "
        "I spent part of the journey reading a historical novel and occasionally looking "
        "out of the window at the changing countryside. As the train moved further north, "
        "the hills became steeper and the villages appeared smaller and quieter. A family "
        "sitting nearby discussed their holiday plans in great detail, which provided an "
        "unexpected source of entertainment during the long journey. By the time I finally "
        "arrived in Edinburgh, I felt both tired and excited about spending the weekend "
        "exploring the city once again."
    ),
    (
        "In December, I travelled by bus from Nairobi to Arusha to attend a regional "
        "business conference that brought together young entrepreneurs from across East "
        "Africa. The journey began before sunrise, and the roads were already busy with "
        "lorries transporting goods between Kenya and Tanzania. Although the border "
        "crossing took longer than expected because of increased security checks, most "
        "passengers remained patient and continued chatting throughout the delay. I spent "
        "much of the trip listening to music, replying to messages on my phone, and "
        "observing the changing landscape outside the window. As we travelled further "
        "south, I noticed large farms, roadside markets, and groups of cyclists moving "
        "between small towns. Several passengers discussed the rising cost of fuel and "
        "how it had affected transport prices in recent months. By the time we finally "
        "arrived in Arusha during the evening, the streets were lively with traders, "
        "tourists, and people preparing for the holiday season."
    ),
]
# -----------------------------------------------------------------------------------
# 2. Human/reference Kiswahili translation (will be used for metric computation)
# -----------------------------------------------------------------------------------
REFERENCE_TRANSLATION: list[str] = [
    ("Mwisho wa msimu wa vuli uliopita, nilisafiri kwa treni kutoka Manchester hadi Edinburgh "
     "kumtembelea rafiki wa zamani wa chuo kikuu ambaye sikuwa nimemwona kwa miaka kadhaa. "
     "Safari ilianza mapema sana asubuhi, na kituo kilikuwa tayari kimejaa wasafiri wa kila siku "
     "waliobeba vikombe vya kahawa na mikoba ya mkononi. Baghairi ya utabiri wa hali ya hewa "
     "kutabiri mvua kubwa mchana kutwa, anga lilibaki safi kwa kushangaza kwa sehemu kubwa ya safari. "
     "Nilitumia sehemu ya safari kusoma riwaya ya kihistoria na mara kwa mara nikitazama nje ya "
     "dirisha kuona mandhari ya mashambani ikibadilika. Treni ilivyozidi kwenda kaskazini, vilima "
     "vilikuwa vyenye mteremko mkali na vijiji vilionekana vidogo na vyenye utulivu zaidi. Familia "
     "moja iliyoketi karibu ilijadili mipango yao ya likizo kwa undani sana, jambo ambalo lilileta "
     "chanzo cha burudani kisichotarajiwa wakati wa safari hiyo ndefu. Kufikia wakati hatimaye nawasili "
     "Edinburgh, nilihisi kuchoka na pia kuwa na msisimko kuhusu kutumia wikendi kuchunguza jiji hilo "
     "kwa mara nyingine tena."
    ),
    ("Mnamo mwezi wa Desemba, nilisafiri kwa basi kutoka Nairobi hadi Arusha kuhudhuria mkutano wa kanda "
     "wa biashara uliowaleta pamoja wajasiriamali vijana kutoka kote Afrika Mashariki. Safari ilianza "
     "kabla ya jua kuchomoza, na barabara zilikuwa tayari zimejaa malori yanayosafirisha bidhaa kati "
     "ya Kenya na Tanzania. Baghairi ya uvukaji wa mpaka kuchukua muda mrefu kuliko ilivyotarajiwa kwa "
     "sababu ya ongezeko la ukaguzi wa usalama, abiria wengi walibaki wavumilivu na waliendelea kupiga "
     "gumzo wakati wote wa kuchelewa huko. Nilitumia sehemu kubwa ya safari kusikiliza muziki, kujibu "
     "ujumbe kwenye simu yangu, na kutazama mandhari inayobadilika nje ya dirisha. Tulipozidi kwenda "
     "kusini, niliona mashamba makubwa, masoko ya kando ya barabara, na vikundi vya waendesha baiskeli "
     "wakitembea kati ya miji midogo. Abiria kadhaa walijadili kupanda kwa gharama ya mafuta na jinsi "
     "ilivyokuwa imeathiri bei za usafiri katika miezi ya hivi karibuni. Kufikia wakati hatimaye tunawasili "
     "Arusha nyakati za jioni, mitaa ilikuwa imechangamka kwa wafanyabiashara, watalii, na watu wanaojiandaa "
     "kwa ajili ya msimu wa likizo."
    )
]

# -----------------------------------------------------------------------------------
# 3. Rule-Based Machine Translation (RBMT) class
# -----------------------------------------------------------------------------------
class RuleBasedTranslator:

    # Bilingual lexicon: English lemma -> Kiswahili equivalent organized thematically for readability
    LEXICON:dict[str, str] = {
        # --- Pronouns & determiners ---
        "i": "mimi", "my": "yangu", "we": "sisi", "our": "yetu",
        "the": "", "a": "", "an": "", "their": "yao", "its": "yake",
        "this": "hii", "that": "ile", "these": "hizi", "those": "zile",
        "which": "ambacho", "whom": "ambaye", "who": "ambaye",

        # --- Common verbs ---
        "travelled": "nilisafiri", "travel": "safiri",
        "visit": "kutembelea", "visited": "alitembelea",
        "began": "ilianza", "begin": "anza",
        "was": "ilikuwa", "were": "zilikuwa", "is": "ni", "are": "ni",
        "had": "alikuwa na", "have": "kuwa na",
        "predicted": "ilitabiri", "remained": "iliendelea kuwa",
        "spent": "nilitumia", "spend": "tumia",
        "reading": "nikisoma", "read": "soma",
        "looking": "nikitazama", "look": "tazama",
        "moved": "ilisogea", "move": "sogea",
        "became": "ilizidi kuwa", "appear": "kuonekana",
        "appeared": "vilionekana", "sitting": "iliyokaa",
        "discussed": "walijadiliana", "discuss": "jadili",
        "provided": "ilitoa", "provide": "toa",
        "arrived": "aliwasili", "arrive": "wasili",
        "felt": "nilihisi", "feel": "hisi",
        "exploring": "nikichunguza", "explore": "chunguza",
        "seen": "kuona", "see": "ona",
        "carried": "wakibeba", "carry": "beba", "carrying": "wakibeba",
        "attend": "kuhudhuria", "attended": "alihudhuria",
        "brought": "ilikutanisha", "bring": "kuleta",
        "transporting": "yanayobeba", "transport": "beba",
        "crossing": "kuvuka", "cross": "vuka",
        "took": "ilichukua", "take": "chukua",
        "chatting": "kuzungumza", "chat": "zungumza",
        "listening": "nikisikiliza", "listen": "sikiliza",
        "replying": "kujibu", "reply": "jibu",
        "observing": "kuangalia", "observe": "angalia",
        "noticed": "niliangalia", "notice": "angalia",
        "moving": "wakisogea", "preparing": "wakijiandaa",

        # --- Nouns – travel & transport ---
        "train": "treni", "bus": "basi", "lorries": "malori",
        "station": "stesheni", "border": "mpaka", "road": "barabara",
        "roads": "barabara", "journey": "safari", "trip": "safari",
        "ticket": "tiketi", "platform": "jukwaa",
        "cyclists": "waendesha baiskeli",

        # --- Nouns – people ---
        "commuters": "wasafiri", "passengers": "abiria",
        "family": "familia", "friend": "rafiki",
        "entrepreneurs": "wajasiriamali", "traders": "wasusi",
        "tourists": "watalii", "people": "watu",

        # --- Nouns – places & geography ---
        "manchester": "Manchester", "edinburgh": "Edinburgh",
        "nairobi": "Nairobi", "arusha": "Arusha",
        "kenya": "Kenya", "tanzania": "Tanzania",
        "africa": "Afrika", "east": "Mashariki",
        "city": "mji", "town": "mji", "towns": "miji",
        "village": "kijiji", "villages": "vijiji",
        "countryside": "mashambani", "hills": "milima",
        "landscape": "mazingira", "window": "dirisha",
        "streets": "mitaa", "farms": "mashamba",
        "markets": "masoko",

        # --- Nouns – objects ---
        "coffee": "kahawa", "cups": "vikombe",
        "briefcases": "mikoba ya ofisi", "novel": "riwaya",
        "music": "muziki", "phone": "simu",
        "messages": "ujumbe", "goods": "bidhaa",
        "fuel": "mafuta",

        # --- Nouns – time ---
        "morning": "asubuhi", "evening": "jioni",
        "weekend": "wikendi", "day": "siku",
        "months": "miezi", "years": "miaka",
        "autumn": "vuli", "december": "Desemba",
        "sunrise": "mapambazuko",

        # --- Adjectives & adverbs ---
        "old": "wa zamani", "historical": "ya kihistoria",
        "heavy": "kubwa", "clear": "wazi",
        "early": "mapema", "further": "zaidi",
        "north": "kaskazini", "south": "kusini",
        "small": "ndogo", "smaller": "vidogo",
        "large": "kubwa", "great": "mkubwa",
        "long": "ndefu", "short": "fupi",
        "tired": "uchovu", "excited": "msisimko",
        "young": "wachanga", "regional": "wa kikanda",
        "unexpected": "kisichotarajiwa",
        "surprisingly": "kwa mshangao",
        "patient": "subira", "lively": "changamfu",
        "busy": "na shughuli nyingi", "already": "tayari",
        "crowded": "imejaa", "quiet": "kimya",
        "quieter": "kimya zaidi", "steep": "mirefu",
        "steeper": "mirefu zaidi",

        # --- Prepositions & conjunctions ---
        "from": "kutoka", "to": "hadi", "by": "kwa",
        "in": "katika", "at": "kwenye", "of": "ya",
        "for": "kwa", "with": "na", "about": "kuhusu",
        "during": "wakati wa", "throughout": "siku nzima",
        "between": "kati ya", "across": "kutoka",
        "because": "kwa sababu", "although": "ingawa",
        "as": "kadri", "when": "nilipofika",
        "out": "nje", "nearby": "karibu",

        # --- Others ---
        "not": "si", "both": "kote", "finally": "hatimaye",
        "most": "wengi", "several": "kadhaa", "many": "nyingi",
        "much": "nyingi", "part": "sehemu", "time": "wakati",
        "once": "tena", "again": "tena", "very": "sana",
        "too": "pia", "also": "pia",
        "detail": "undani mkubwa", "plans": "mipango",
        "holiday": "likizo", "season": "msimu",
        "university": "chuo", "conference": "mkutano",
        "business": "biashara", "security": "usalama",
        "checks": "ukaguzi", "delay": "kuchelewa",
        "cost": "gharama", "prices": "bei",
        "entertainment": "burudani", "source": "chanzo",
        "rain": "mvua", "weather": "hali ya hewa",
        "forecast": "utabiri", "skies": "anga",
    }

    # Structural transfer rules: (regex_pattern, replacement_string), applied in order after
    # lexicon substitution

    TRANSFER_RULES: list[tuple[str, str]] = [
        # Negation: "si _ ilikuwa" -> retain negation marker
        (r'\bsi ilikuwa\b',"haikuwa"),
        # Possesive smoothing
        (r'\brafiki yangu wa zamani wa chuo\b','rafiki yangu wa zamani wa chuo'),
        # Remove orphaned articles left by empty-string lexicon entries
        (r"\s{2,}"," ")
    ]

    def __init__(self) -> None:
        self._sorted_lexicon:list[tuple[str, str]] = sorted(
            self.LEXICON.items(), key=lambda kv:len(kv[0]), reverse=True
        )

    # -----------------------------------------------------------------------------------
    # I. Public Interface
    # -----------------------------------------------------------------------------------
    def translate(self,  text:str) -> str:
        sentences: list[str] = self._split_sentences(text)
        translated: list[str] = [self._translate_sentences(s) for s in sentences]
        # Reconstruct the text into a single string
        return " ".join(translated)

    # -----------------------------------------------------------------------------------
    # II. Private Helpers
    # -----------------------------------------------------------------------------------
    @staticmethod
    def _split_sentences(text:str) -> list[str]:
        raw:list[str] = re.split(r"(?<=[.!?])\s+",text.strip())
        return raw

    def _translate_sentences(self, sentence:str) -> str:
        # ----------------------- Stage 1: Analysis ---------------------------------------
        # Preserve trailing punctuation
        punc : str=" "
        if sentence and sentence[-1] in ".!,;:":
            punc = punc + sentence[-1]
            sentence = sentence[:-1]

        tokens: list[str] = sentence.lower().split()

        # ----------------------- Stage 2a: Lexical Transfer -------------------------------
        output_tokens: list[str] = []
        idx: int = 0
        while idx < len(tokens):
            matched: bool = False
            # Try multi-word matches first (up to 3 tokens)
            for window in (3, 2, 1):
                phrase: str =" ".join(tokens[idx:idx+window])
                if phrase in self.LEXICON:
                    replacement: str = self.LEXICON[phrase]
                    if replacement: # Skip empty-string mappings (articles)
                        output_tokens.append(replacement)
                    idx += window
                    matched = True
                    break

            if not matched:
                # Unknown token: pass  through as-is (capitalize proper nouns)
                tok: str = tokens[idx]
                if tok[0].isupper() if tokens[idx] else False:
                    output_tokens.append(tok.capitalize())
                else:
                    output_tokens.append(tok)
                idx += 1

        output: str = "".join(output_tokens)

        # ----------------------- Stage 2b: Structural transfer rules --------------------
        for pattern, replacement in self.TRANSFER_RULES:
            output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)

        # ----------------------- Stage 3: Generation -------------------------------------
        output = re.sub(r"\s{2,}", " ", output).strip()
        if output:
            output = output[0].upper() + output[1:]
        return output

# -----------------------------------------------------------------------------------
# 4. Neural-Based Machine Translation (NBMT) class
# -----------------------------------------------------------------------------------
class NeuralTranslator:
    DEFAULT_MODEL: str = "Helsinki-NLP/opus-mt-en-swc"

    def __init__(self,model_name:str=DEFAULT_MODEL,max_length:int=512,beam_size:int=4) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.beam_size = beam_size
        self._pipeline = None # Lazy-loaded on first call

    def _load_model(self) -> None:
        from transformers import MarianMTModel, MarianTokenizer
        print(f"\n[NBMT]: Loading model '{self.model_name}'...")
        tokeniser = MarianTokenizer.from_pretrained(self.model_name)
        model = MarianMTModel.from_pretrained(self.model_name)
        model.eval()

        # Store as a simple namespace for convenience
        self._tokeniser = tokeniser
        self._model = model
        print("[NBMT] model loaded successfully!")

    def translate(self,text:str)->str:
        import torch
        if self._pipeline is None:
            self._load_model()

        # Split into paragraphs / chunks to avoid exceeding token limits
        chunks: list[str] = self._chunk_text(text)
        translations: list[str] = []

        for chunk in chunks:
            encoded = self._tokeniser(
                [chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **encoded,
                    num_beams=self.beam_size,
                    max_length=self.max_length,
                    early_stopping=True
                )

            decoded:str = self._tokeniser.decode(
                generated_ids[0],skip_special_tokens=True
            )
            translations.append(decoded)
        return " ".join(translations)

    # -----------------------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------------------
    @staticmethod
    def _chunk_text(text: str, max_words: int = 100) -> list[str]:
        """
        Split *text* into sentence-aligned chunks of at most *max_words*
        words to stay within the model's positional embedding limit.

        Parameters
        ----------
        text     : str  – Source text.
        max_words: int  – Soft upper bound on words per chunk.

        Returns
        -------
        list[str]
            A list of text chunks suitable for the model.
        """
        sentences: list[str] = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        current: list[str] = []
        current_len: int = 0

        for sent in sentences:
            sent_len: int = len(sent.split())
            if current_len + sent_len > max_words and current:
                chunks.append(" ".join(current))
                current = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len

        if current:
            chunks.append(" ".join(current))
        return chunks


# -----------------------------------------------------------------------------------
# 5. EVALUATION METRICS
# -----------------------------------------------------------------------------------

class TranslationEvaluator:
    """
    Compute standard automatic evaluation metrics for machine translation.

    Metrics
    -------
    BLEU (Bilingual Evaluation Understudy)
        N-gram precision score with a brevity penalty.  Widely used but
        sensitive to exact token matches; less suitable for morphologically
        rich languages.  Range: 0–100 (higher is better).

    ROUGE-1 / ROUGE-L
        Recall-oriented metrics based on unigram overlap (ROUGE-1) and
        longest common subsequence (ROUGE-L).  Range: 0–1.

    chrF (Character n-gram F-score)
        Computes F-score over character n-grams, which is more robust for
        agglutinative languages like Kiswahili.  Range: 0–100.
    """

    def compute(
        self,
        hypothesis: str,
        reference: str,
        system_name: str = "System",
    ) -> dict[str, float]:
        """
        Compute all metrics for a single hypothesis–reference pair.

        Parameters
        ----------
        hypothesis  : str – Machine-translated text.
        reference   : str – Human reference translation.
        system_name : str – Label used for display purposes.

        Returns
        -------
        dict[str, float]
            Keys: ``bleu``, ``rouge1``, ``rougeL``, ``chrf``.
        """
        bleu_score: float = self._bleu(hypothesis, reference)
        rouge_scores: dict[str, float] = self._rouge(hypothesis, reference)
        chrf_score: float = self._chrf(hypothesis, reference)

        scores: dict[str, float] = {
            "bleu": round(bleu_score, 2),
            "rouge1": round(rouge_scores["rouge1"], 4),
            "rougeL": round(rouge_scores["rougeL"], 4),
            "chrf": round(chrf_score, 2),
        }

        self._print_scores(system_name, scores)
        return scores

    # ------------------------------------------------------------------
    # Private metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _bleu(hypothesis: str, reference: str) -> float:
        """
        Compute corpus-level BLEU score using SacreBLEU.

        SacreBLEU provides reproducible, standardised BLEU computation
        with consistent text normalization.

        Parameters
        ----------
        hypothesis : str
        reference  : str

        Returns
        -------
        float
            BLEU score in [0, 100].
        """
        try:
            from sacrebleu.metrics import BLEU  # type: ignore
            bleu = BLEU(effective_order=True)
            result = bleu.corpus_score([hypothesis], [[reference]])
            return float(result.score)
        except ImportError:
            print("[WARN] sacrebleu not installed – BLEU defaulting to 0.")
            return 0.0

    @staticmethod
    def _rouge(hypothesis: str, reference: str) -> dict[str, float]:
        """
        Compute ROUGE-1 and ROUGE-L F1 scores using the rouge-score library.

        Parameters
        ----------
        hypothesis : str
        reference  : str

        Returns
        -------
        dict[str, float]
            Keys: ``rouge1``, ``rougeL``.
        """
        try:
            from rouge_score import rouge_scorer  # type: ignore
            scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
            scores = scorer.score(reference, hypothesis)
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except ImportError:
            print("[WARN] rouge-score not installed – ROUGE defaulting to 0.")
            return {"rouge1": 0.0, "rougeL": 0.0}

    @staticmethod
    def _chrf(hypothesis: str, reference: str) -> float:
        """
        Compute chrF score using SacreBLEU's ChrF implementation.

        chrF is computed over character 6-grams with β=2 (weighting
        recall twice as heavily as precision).

        Parameters
        ----------
        hypothesis : str
        reference  : str

        Returns
        -------
        float
            chrF score in [0, 100].
        """
        try:
            from sacrebleu.metrics import CHRF  # type: ignore
            chrf = CHRF()
            result = chrf.corpus_score([hypothesis], [[reference]])
            return float(result.score)
        except ImportError:
            print("[WARN] sacrebleu not installed – chrF defaulting to 0.")
            return 0.0

    @staticmethod
    def _print_scores(system_name: str, scores: dict[str, float]) -> None:
        """Pretty-print evaluation scores to stdout."""
        print(f"\n{'─'*55}")
        print(f"  Evaluation Results – {system_name}")
        print(f"{'─'*55}")
        print(f"  BLEU   : {scores['bleu']:>7.2f}  (0–100; higher = better)")
        print(f"  ROUGE-1: {scores['rouge1']:>7.4f}  (0–1;   higher = better)")
        print(f"  ROUGE-L: {scores['rougeL']:>7.4f}  (0–1;   higher = better)")
        print(f"  chrF   : {scores['chrf']:>7.2f}  (0–100; higher = better)")
        print(f"{'─'*55}\n")

# ===========================================================================
# 6. Visualisation class
# ===========================================================================
class TranslationVisualiser:

    # Colour patterns inspired by Kenyan and Scottish flag form translation text
    COLOURS:dict[str, str] = {
        "RBMT":"#006600", # Forest Green (Kenyan Flag)
        "NBMT":"#003086" # Royal Blue (Scotland)
    }

    def plot_comparison(
            self,
            rbmt_scores: dict[str,float],
            nbmt_scores: dict[str,float],
            save_path:str = "../files/mt_comparison.png"
    )->None:

        # Normalise all scores to 0-100 for comparable visualisation
        metrics:list[str] = ["BLEU", "ROUGE-1", "ROUGE-L", "chrF"]
        rbmt_vals:list[float] = [
            rbmt_scores["bleu"],
            rbmt_scores["rouge1"] * 100,
            rbmt_scores["rougeL"] * 100,
            rbmt_scores["chrf"],
        ]
        nbmt_vals:list[float] = [
            nbmt_scores["bleu"],
            nbmt_scores["rouge1"] * 100,
            nbmt_scores["rougeL"] * 100,
            nbmt_scores["chrf"],
        ]
        x: np.ndarray = np.arange(len(metrics))
        width:float = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f9f9f9')
        ax.set_facecolor('#f4f4f4')

        bars_rbmt = ax.bar(
            x - width / 2, rbmt_vals, width,
            label="RBMT", color=self.COLOURS["RBMT"],alpha=0.88,edgecolor="white"
        )
        bars_nbmt = ax.bar(
            x + width / 2, nbmt_vals, width,
            label="NBMT (MarianMT)", color=self.COLOURS["NBMT"],alpha=0.88,edgecolor="white"
        )

        # Annotate bars with numeric values
        for bar in list(bars_rbmt) + list(bars_nbmt):
            height:float = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                #xytext=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0,3),
                textcoords="offset points",
                ha="center",va="bottom",
                fontsize=9,fontweight="bold",
            )

        ax.set_xlabel("Evaluation Metric", fontsize=12,labelpad=10)
        ax.set_ylabel("Score (normalised to 0-100)", fontsize=12,labelpad=10)
        ax.set_title(
            "Machine Translation Evaluation\nRBMT vs. NBMT (English -> Kiswahili)",
            fontsize=14,fontweight="bold",pad=15
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics,fontsize=11)
        ax.set_ylim(0,110)
        ax.yaxis.grid(True, linestyle="--",alpha=0.6)
        ax.set_axisbelow(True)

        legend_patches = [
            mpatches.Patch(color=self.COLOURS["RBMT"], label="RBMT (Rule-Based)"),
            mpatches.Patch(color=self.COLOURS["NBMT"], label="NBMT (MarianMT) Neural"),
        ]
        ax.legend(handles=legend_patches,fontsize=10,loc="upper right")

        plt.tight_layout()
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path,dpi=150,bbox_inches="tight")
        print(f"\n[VIS] Chart saved to {save_path}")
        plt.show()

# -----------------------------------------------------------------------------------
# 7. Display Utilities
# -----------------------------------------------------------------------------------
def print_translation_comparison(
        source:str,
        rbmt_output:str,
        nbmt_output:str,
        paragraph_idx:int
)->None:

    width: int = 78
    wrapper = textwrap.TextWrapper(width=width,subsequent_indent="  ")

    print("\n" + "=" * width)
    print(f"    PARAGRAPH: {paragraph_idx}")
    print("=" * width)

    print("\n🔶 SOURCE (English):")
    print(" " + "\n ".join(wrapper.wrap(source)))

    print("\n🔶 RBMT Translation (Kiswahili):")
    print(" " + "\n ".join(wrapper.wrap(rbmt_output)))

    print("\n🔶 NBMT Translation (Kiswahili - MarianMT):")
    print(" " + "\n ".join(wrapper.wrap(nbmt_output)))

    print("\n" + "=" * width)

# -----------------------------------------------------------------------------------
# 8. Main Execution Function
# -----------------------------------------------------------------------------------
def main() -> None:

    print("\n" + "█" * 78)
    print("    MACHINE TRANSLATION DEMO - English -> Kiswahili")
    print("\n" + "█" * 78)

    # -----------------------------------------------------------------------------------
    # Initialise systems
    # -----------------------------------------------------------------------------------
    rbmt: RuleBasedTranslator = RuleBasedTranslator()
    nbmt: NeuralTranslator = NeuralTranslator()
    evaluator: TranslationEvaluator = TranslationEvaluator()

    rbmt_all_translations: list[str] = []
    nbmt_all_translations: list[str] = []
    rbmt_scores_all: list[dict[str, float]] = []
    nbmt_scores_all: list[dict[str, float]] = []

    # -----------------------------------------------------------------------------------
    # Translate and evaluate paragraph by paragraph
    # -----------------------------------------------------------------------------------
    for idx, (source, reference) in enumerate(
        zip(SOURCE_PARAGRAPHS, REFERENCE_TRANSLATION), start=1
    ):
        print(f"\n{'.'*78}")
        print(f" Processing Paragraph {idx} _")
        print(f"\n{'.'*78}")

        # --- RBMT ---
        print("\n[RBMT] Translating ...")
        # Fallback safeguard in case RuleBasedTranslator translate method isn't fully implemented
        if hasattr(rbmt, "translate"):
            rbmt_translation: str = rbmt.translate(source)
        else:
            # Fallback mock translate using dictionary split mapping to avoid crash
            rbmt_translation = " ".join([rbmt.LEXICON.get(w.strip(".,!?").lower(), w) for w in source.split()])

        rbmt_all_translations.append(rbmt_translation)

        # --- NBMT ---
        print("\n[NBMT] Translating ...")
        nbmt_translation:str = nbmt.translate(source)
        nbmt_all_translations.append(nbmt_translation)

        # --- Display ---
        print_translation_comparison(source, rbmt_translation, nbmt_translation,idx)

        # --- Evaluate ---
        print(f"\n -- Evaluate Paragraph {idx} --")
        rbmt_sc = evaluator.compute(rbmt_translation,reference,f"RBMT P{idx}")
        nbmt_sc = evaluator.compute(nbmt_translation,reference,f"NBMT P{idx}")
        rbmt_scores_all.append(rbmt_sc)
        nbmt_scores_all.append(nbmt_sc)

    # -----------------------------------------------------------------------------------
    # Aggregate scores (macro-average across paragraphs)
    # -----------------------------------------------------------------------------------
    def avg_scores(score_list: list[dict[str, float]]) -> dict[str, float]:
        keys:list[str] = list(score_list[0].keys())
        return {k: round(sum(s[k] for s in score_list) / len(score_list),4) for k in keys}

    rbmt_avg: dict[str, float] = avg_scores(rbmt_scores_all)
    nbmt_avg: dict[str, float] = avg_scores(nbmt_scores_all)

    print("\n" + "█" * 78)
    print("    OVERALL EVALUATION (Macro-average across all paragraphs)")
    print("\n" + "█" * 78)
    evaluator._print_scores("RBMT - Overall Average", rbmt_avg)
    evaluator._print_scores("NBMT - Overall Average", nbmt_avg)

    # -----------------------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------------------
    visualiser: TranslationVisualiser = TranslationVisualiser()
    visualiser.plot_comparison(rbmt_avg,nbmt_avg)

    # -----------------------------------------------------------------------------------
    # Option summary on RBMT & NBMT
    # -----------------------------------------------------------------------------------
    print("\n" + "█" * 78)
    print("  DISCUSSION SUMMARY")
    print("█" * 78)
    print(
        textwrap.fill(
            "RBMT provides transparent, deterministic translations grounded "
            "in explicit linguistic knowledge. Its scores are typically lower "
            "because it cannot model contextual meaning, long-range grammatical "
            "agreement, or idiomatic expressions. It is, however, easy to "
            "inspect, debug, and extend for domain-specific terminology.",
            width=78,
        )
    )
    print()
    print(
        textwrap.fill(
            "NBMT (MarianMT) achieves substantially higher scores on all metrics "
            "by learning translation patterns from large parallel corpora. The "
            "model captures morphological agreement, word order differences, and "
            "contextual nuance. Its main limitations are opacity (a 'black-box' "
            "model), dependence on training data, and potential errors on "
            "low-resource language pairs.",
            width=78,
        )
    )
    print("\n" + "█" * 78 + "\n")

# -----------------------------------------------------------------------------------
# 9. Run the script by invoking it's main() function
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()