# Embedding Agentic Traces: Literature Context

*Helivan Research — August 2026. Companion to `NARRATIVE.md`, `FINDINGS.md`, and `PAPER.md`. Part I is a draft related-work narrative positioning the project; Part II is the annotated bibliography it draws on, organized by strand. Citations below were located and verified against arXiv / ACL Anthology / publisher pages during the August 2026 sweep; re-verify identifiers before camera-ready (a handful of 2026 preprints may migrate venues or versions).*

## Part I — Related-Work Narrative (draft)

### 1. Every embedding chooses an equivalence relation; ours makes the choice explicit

The intellectual history of document representation is a history of implicit choices about what "similar" means. Salton's vector space model (1975) made the founding move — a document is a point, similarity is proximity — with term overlap as the implicit equivalence relation, and Sparck Jones's IDF (1972) already encoded the recognition that raw shared signal misleads and must be discounted against corpus-wide statistics. Every subsequent generation rediscovered, in its own vocabulary, that the dominant variance of a learned representation is often nuisance rather than meaning. LSA (Deerwester et al., 1990) freed similarity from lexical overlap but surrendered control over what the latent dimensions encode; LDA's topics (Blei et al., 2003) notoriously resolve to sources, authors, and boilerplate in heterogeneous corpora. In the neural era the same lesson recurs geometrically: word and sentence embedding spaces carry a large common component and a few dominant directions encoding frequency and style rather than content, and simply removing them helps almost everything (SIF, Arora et al. 2017; All-but-the-Top, Mu et al. 2018); contextual spaces are anisotropic cones in which the shared component swamps content differences (Ethayarajh 2019; BERT-flow, Li et al. 2020; rogue dimensions, Timkey & van Schijndel 2021). Probing studies formalized the diagnosis: a single vector entangles length, word order, surface form, and topic, and the training objective — not the user's intent — determines the mix (Adi et al. 2017; Conneau et al. 2018). Meanwhile the stylometry literature established the mirror-image fact that authorship is one of the strongest and most learnable surface signals in text (Stamatatos 2009; LUAR, Rivera-Soto et al. 2021), and that style and content contaminate each other's representations unless explicitly controlled (Wegmann et al. 2022).

The historical fixes fall into three families: post-hoc removal of the shared component (IDF, SIF, ABTT, whitening), restructured aggregation (hierarchical pooling, Yang et al. 2016; SWEM, Shen et al. 2018; section- and aspect-wise encoding, SPECTER/Aspire), and changed supervision that encodes the intended relation (paraphrase pairs, Wieting et al. 2016; NLI, InferSent/SBERT; citations, SPECTER). Our method can be read as the LLM-era unification of all three at once: the rubric is supervision-by-instruction (the intended equivalence relation written in natural language rather than learned from labeled pairs), per-section extraction and embedding is semantically indexed hierarchical pooling, and consensus centering is common-component removal generalized from a global frequency confound to a per-instance authorship confound. What is new is the modality that forces the issue: agentic traces are documents whose authorship signal is more dominant, higher-rank, and more structural (tool syntax, action cadence, harness boilerplate) than anything the classical literature faced.

### 2. Modern embedders and the limits of one fixed similarity

The modern pipeline — contrastively trained bi-encoders from DPR and Contriever through E5, GTE, BGE, and the LLM-backbone generation (E5-Mistral, NV-Embed, LLM2Vec, GritLM, Gecko, Gemini Embedding, Qwen3-Embedding) — produces a single similarity function fixed at training time and ranked on MTEB. Both theory and benchmark politics now concede this is not enough: Weller et al. (2025) prove via sign-rank arguments that no single fixed d-dimensional space can represent all relevance relations (the LIMIT construction), and the benchmark ecosystem's own responses (MMTEB's overfitting concerns, RTEB's private held-out sets) acknowledge that public leaderboard scores overstate faithfulness off-distribution. Long-context patches (LongEmbed, late chunking, BGE-M3, ColBERT-style late interaction) address how much text a model can ingest, not which aspects of it drive similarity — and for traces, token-level matching actively amplifies the harness boilerplate we need to ignore. MTEB contains no task in which the criterion of similarity varies per query; our setting makes that variation the central object.

### 3. Programmable similarity: the closest prior art

A vigorous 2022–2026 strand loosens the fixed-similarity assumption by conditioning. Instruction-tuned embedders (Instructor, Su et al. 2022; TART, Asai et al. 2022; task-instruction conventions in E5-Mistral and Qwen3; adapter-conditioned Jina v3; format-conditioned SPECTER2) let a task tag select among trained-in notions of similarity. Promptriever (Weller et al. 2024) shows retrievers can honor free-form per-query definitions of relevance; C-STS (Deshpande et al. 2023) and its successors (Tu et al. 2024; Hyper-CL; CASE) formalize similarity conditioned on a stated aspect and show SOTA embedders and GPT-4 handle it poorly; the instruction-following-retrieval benchmarks (FollowIR, InstructIR, IFIR, MAIR) document that trained-in instruction following is shallow and breaks off-distribution. Closest of all mechanically are the QA-mediated representations: HyDE embeds LLM-generated text in place of the query; InBedder (2024) embeds a model's answers to an instruction-as-question; QA-Emb (Benara et al. 2024) and CQG-MBQA (2024) represent a text by its answers to a battery of natural-language questions; Dense X Retrieval re-structures documents into LLM-extracted propositions before embedding; and Ravfogel et al.'s description-based similarity (2023) states our motivating philosophy — similarity is underspecified until the notion of sameness is — and operationalizes one fixed notion of it.

Every located method in this strand, however, does at least one of the following: operates on sentences or short passages; accepts a single short instruction rather than a structured, per-instance, multi-section rubric; bakes the honored instruction distribution into training; or produces judgments and labels rather than a reusable vector space. None addresses invariance to the system that authored the text — the dominant nuisance in multi-system trace corpora. Qubric occupies the unclaimed intersection: inference-time, training-free, rubric-specified similarity over very long agentic artifacts, with an explicit invariance mechanism (consensus centering) enforcing the "invariant to authorship" half of the specification.

### 4. Why not erase authorship geometrically?

The invariance literature offers three alternative families, all operating within a fixed representation. Unsupervised geometric post-processing (ABTT, whitening, BERT-flow) reshapes global variance but cannot target authorship specifically, and bijective maps provably preserve all information. Supervised concept erasure — INLP (Ravfogel et al. 2020), R-LACE, kernelized erasure, LEACE (Belrose et al. 2023), and, nearest to our problem, Fan et al.'s (2025) LEACE-style deconfounding of document embeddings by source — removes a labeled nuisance, but its own literature documents chronic incompleteness: post-hoc classifiers recover "removed" attributes (Elazar & Goldberg 2018), residual cluster structure survives projection (Gonen & Goldberg 2019), linear guarantees do not bind nonlinear or downstream adversaries (Ravfogel et al. 2022, 2023), and closing nonlinear leaks costs task utility (Obliviator, 2026) — a tension Zhao & Gordon (2019) prove is inherent when invariance and sufficiency compete inside one space. Invariant representation learning (DANN, censored representations, VFAE, Moyer et al. 2018) needs domain labels and retraining and inherits the same leakage–utility frontier. The fingerprinting results (below) explain why traces sit in the worst regime for all three: the authorship signal is robust to paraphrase, pervades action structure as well as wording, and is plausibly high-rank and nonlinear — precisely where projection either leaks or destroys content. Our approach is categorically different: rather than subtracting authorship from a fingerprint-rich signal, it changes the information source — a judge re-observes the trace through a content rubric, emitting descriptions in which authorship style is largely never encoded, and residual judge- and phrasing-level idiosyncrasy is cancelled by centering against the cross-system consensus (a cohort-level relative of common-component removal and CSLS neighborhood centering). Invariance by reconstruction from a fingerprint-poor channel, not by surgery on a fingerprint-rich one.

### 5. Fingerprints are real, robust, and — until now — only an attacker's asset

Our raw-embedding finding (exact agent configuration identifiable at twice chance even among same-model, same-harness siblings) lands in a fast-moving attribution literature. Model-level fingerprints are pervasive and survive paraphrase, translation, and summarization (Idiosyncrasies, Sun et al. 2025); they are extractable watermark-free (Sniffer, 2023), persist across languages (2025), organize models into recoverable phylogenies ("LLM DNA," ICLR 2026), and appear in generated code with style separable from semantics (Code Fingerprints, 2026). In 2026 the phenomenon reached the agent level: coding-agent trajectories attribute to scaffold and model at 85.7% vs 11% chance (Oderinwale 2026), browser agents are identified from UI action traces at up to 96% F1, and web-agent frameworks are separable from network and interaction signatures at 97%. Surveys through 2024 contain no treatment of harness-level attribution — the sub-field is months old, and all of it is attacker-side. To our knowledge no prior work treats the fingerprint as a nuisance to be removed from trace representations; our inverse use of raw embeddings as provenance detectors is the constructive reading of the same measurement. Judge-side, self-recognition and self-preference (Panickssery et al. 2024; Wataoka et al. 2024) and preference leakage across model families (Li et al. 2025) are the same phenomenon appearing inside the evaluation loop — the mechanistic basis for our leave-one-model-out protocol.

### 6. Rubrics and judges: mature machinery, repurposed output

The judge stage of qubric inherits a mature literature. LLM-as-judge is established (MT-Bench, Zheng et al. 2023; G-Eval) along with its bias catalog — position (Wang et al. 2023), verbosity (Saito et al. 2023), self-preference, and family leakage — which predicts exactly the two trace-level effects we measure (verbosity as mediator; family leakage inflating naive evaluation). Instance-specific criteria are now the reliable interface to LLM judgment: physician-written per-conversation rubrics at scale (HealthBench, 2025), LLM-generated instruction-specific checklists that measurably raise judge agreement (TICK, CheckEval), rubrics as RL rewards (RLCF; Rubrics-as-Rewards), and 2026 work on automatic rubric generation and its meta-evaluation (RubricRAG, RubricEval, dynamic rubric refinement). Trajectory-level judging exists for agents (Agent-as-a-Judge, Zhuge et al. 2024; AgentRewardBench; WebJudge; AgentLens), and the reliability literature (JUDGE-BENCH; Rating Roulette; TRAIL's finding that frontier models localize errors in long traces at ~11% accuracy) predicts our measured per-trace judge disagreement. The delta is constant across all of it: every one of these systems outputs a score; qubric uses the rubric as a representation schema, and no prior work reports judge agreement as a function of aggregation over instances — the per-trace-unreliable / per-agent-reliable redemption curve — for any trace-derived quantity.

### 7. Traces as objects of study

Trace analysis beyond pass/fail became an active area in 2025–2026: symbolic encodings of 9K+ leaderboard trajectories (Mehtiyev & Assunção 2026), canonical action taxonomies with per-scaffold adapters that explicitly document format fragmentation (TraceProbe 2026), success/failure trajectory studies (Majgaonkar et al. 2025), single-scaffold model comparisons at 138K trajectories (Gupta et al. 2026), failure taxonomies with LLM annotation pipelines (MAST, κ≈0.88), and interactive transcript tooling (Transluce's Docent). These produce taxonomies, statistics, or free-text reviews — not reusable vector representations — and none addresses the authorship dominance their own fingerprinting sibling demonstrates. Adjacent embedding work exists for short dialogues (dial2vec), RL state-action trajectories (Ge et al. 2025), and retrieval-for-reuse of an agent's own past traces (ExpeL, InsightEmb, Retrieval-of-Thought), but always on short interactions or for the agent's own benefit. The ecosystem literature (SWE-bench and Verified; SWE-agent; OpenHands; UTBoost's finding that a quarter of Verified leaderboard entries contain mislabeled patches; the SWE-Bench Illusion memorization results; Kapoor et al.'s cost critique; HAL's $40K-per-sweep accounting and 2.5B-token trace release) simultaneously supplies our corpus, motivates distrust of the single pass/fail bit, and quantifies the evaluation cost our payoff attacks.

### 8. Query-efficient benchmarking: richer probes, not just fewer items

Efficient evaluation has converged on one structural fact: the model×instance success matrix over an LLM population is approximately low-rank, so full scores are predictable from few observations — via curated anchor items (Anchor Points; tinyBenchmarks), IRT and adaptive testing (metabench, ATLAS, amortized IRT), sparse optimization (SparseEval 2026), proxy task batteries (Pace 2026), and agent-specific difficulty filtering (Ndzomga 2026). But these methods consume only correctness patterns — at most a few bits per probe — which produces a documented resolution floor (micro-benchmarks cannot separate models within ~3.5–4 points at extreme budgets; Yauney et al. 2025) and absolute-score collapse under scaffold shift. Correctness-based model embeddings (EmbedLLM, IRT-Router, IrtNet, LOCUS) and observational scaling laws (Ruan et al. 2024) embed models by score vectors; REP-CORE (2026) shows richer-than-correctness signals help but requires white-box hidden states; Agent Psychometrics (2026) adds task content on our exact benchmark. Our machinery is the DKPS lineage (data kernels, Duderstadt et al. 2023; DKPS inference, Helm et al. ACL 2025; consistency and concentration theory, Acharyya et al.), and the closest prior work anywhere is Helm, Johnson & Priebe (ICML 2026): predicting a new model's score by comparing few probe responses against cached reference coverage, blended with the sample score. Our contribution is the agentic generalization — each probe is an entire trace, represented by qubric rather than raw response text, with product kernels, representation fusion, greedy probe selection, and a leave-one-model-family-out protocol that the family-correlation literature (Great Models Think Alike, 2025; LOCUS's family clustering) shows is necessary rather than conservative. A single trace carries far more than one bit about how a system works; that is why one probe reaches accuracy that correctness-only methods need eight to ten scored runs to match.

### 9. The gap, stated once

Assembled: (i) no prior work embeds long, heterogeneous agent traces from incompatible third-party formats into a common space; (ii) no prior work uses LLM-generated instance-specific rubrics as a representation schema rather than a scoring instrument; (iii) no prior work targets authorship invariance in trace representations — the entire agent-fingerprinting literature is attacker-side; (iv) no prior work jointly quantifies judge reliability, agent stochasticity, and embedder stability for trace-derived representations; and (v) no prior work predicts full-benchmark agent performance from the content of probe traces rather than correctness patterns. Each neighboring strand supplies a piece — conditioned similarity, consensus centering, rubric judging, perspective-space inference — and the combination is where this project is first.

## Part II — Annotated Bibliography by Strand

*Format: citation — contribution — relevance to this project. Entries verified against primary sources August 2026.*

### A. The historical lineage of text and document embeddings

#### A.1 Vector-space IR and latent-variable document semantics

- Sparck Jones (1972). "A Statistical Interpretation of Term Specificity…" J. Documentation 28(1). Introduces IDF. — The first formal statement that shared/ubiquitous signal misleads and must be discounted against corpus statistics; the earliest ancestor of consensus centering.
- Salton, Wong & Yang (1975). "A Vector Space Model for Automatic Indexing." CACM 18(11). Documents as points, similarity as proximity. — The founding move; qubric interrogates the equivalence relation Salton chose implicitly (term overlap = similarity).
- Deerwester et al. (1990). "Indexing by Latent Semantic Analysis." JASIS 41(6). Truncated SVD of the term–document matrix. — Origin of dense document embeddings, and of the loss of control over what latent dimensions encode.
- Hofmann (1999). "Probabilistic Latent Semantic Indexing." SIGIR 1999. LSA as a latent-variable model. — Documents as posteriors over interpretable factors; rubric sections are the hand-specified, LLM-era version.
- Blei, Ng & Jordan (2003). "Latent Dirichlet Allocation." JMLR 3. — Classic demonstration that unsupervised document factors are whatever explains co-occurrence variance; in heterogeneous corpora, "topics" resolve to sources and authors — the ancestor of authorship dominance.

#### A.2 Word embeddings

- Mikolov et al. (2013). "Efficient Estimation of Word Representations…" (word2vec). arXiv:1301.3781; and "Distributed Representations of Words and Phrases…" NeurIPS 2013, arXiv:1310.4546. — Substrate of all later embeddings; negative sampling is the ur-form of contrastive learning; frequent-word subsampling is an early "suppress the shared signal" mechanism.
- Pennington, Socher & Manning (2014). "GloVe." EMNLP 2014. Global co-occurrence factorization. — Makes explicit that embeddings are corpus-statistics compressors; the strongest regularity in the corpus (for traces: harness formatting) dominates the geometry.
- Levy & Goldberg (2014). "Neural Word Embedding as Implicit Matrix Factorization." NeurIPS 2014. — Theoretical bridge: neural embeddings inherit exactly the biases of the co-occurrence statistics they factorize; grounds the claim that authorship dominance is a property of distributional objectives, not one encoder.
- Bojanowski et al. (2017). "Enriching Word Vectors with Subword Information" (fastText). TACL 5, arXiv:1607.04606. — Character n-grams are the classic stylometric feature family; one mechanistic route by which surface formatting becomes an authorship fingerprint.

#### A.3 Document and sentence embeddings (pre-Transformer)

- Le & Mikolov (2014). "Distributed Representations of Sentences and Documents" (doc2vec). ICML 2014, arXiv:1405.4053. — Direct ancestor of "embed the whole trace as one vector"; its predict-your-own-words objective guarantees encoding of the generator's lexical idiosyncrasies.
- Dai, Olah & Le (2015). "Document Embedding with Paragraph Vectors." arXiv:1507.07998. — Introduces document-similarity triplet evaluation; ancestor of our content-vs-authorship retrieval probes.
- Lau & Baldwin (2016). "An Empirical Evaluation of doc2vec." RepL4NLP, arXiv:1607.05368. — Early evidence that off-the-shelf document embeddings are fragile and corpus-dependent.
- Kiros et al. (2015). "Skip-Thought Vectors." NeurIPS 2015, arXiv:1506.06726. — Sentence-level distributional hypothesis; applied to traces, "similar contexts ⇒ similar vectors" groups by harness, not task.
- Wieting et al. (2016). "Towards Universal Paraphrastic Sentence Embeddings." ICLR 2016, arXiv:1511.08198. — Early supervised specification of the equivalence relation via paraphrase pairs; qubric replaces pairs with a written rubric.
- Hill, Cho & Korhonen (2016). "Learning Distributed Representations of Sentences from Unlabelled Data." NAACL 2016, arXiv:1602.03483. — First clear articulation that the best representation depends on the target similarity — the thesis statement of programmable similarity, avant la lettre.
- Arora, Liang & Ma (2017). "A Simple but Tough-to-Beat Baseline for Sentence Embeddings" (SIF). ICLR 2017. — Weighted averaging plus removal of the first principal component ("common discourse"); the closest classical analogue of consensus centering, with a generative-model justification.
- Khodak et al. (2018). "A La Carte Embedding." ACL 2018, arXiv:1805.05388. — Precedent for building representations of arbitrary user-defined semantic units atop a fixed space; kin to embedding rubric sections as units.
- Shen et al. (2018). "Baseline Needs More Love" (SWEM). ACL 2018, arXiv:1805.09843. — Pooling strategy matters more than encoder power on long text; motivates structured per-section aggregation over monolithic encoding.

#### A.4 Universal encoders, the BERT failure, and contrastive fixes

- Conneau et al. (2017). "Supervised Learning of Universal Sentence Representations…" (InferSent). EMNLP 2017, arXiv:1705.02364. — The supervision task defines what the embedding encodes; NLI selects for propositional content over surface form.
- Cer et al. (2018). "Universal Sentence Encoder." arXiv:1803.11175. — The moment "off-the-shelf embedding API" became the default workflow — the practice whose failure on traces motivates the paper.
- Peters et al. (2018). "Deep contextualized word representations" (ELMo). NAACL 2018, arXiv:1802.05365; Devlin et al. (2019). "BERT." NAACL 2019, arXiv:1810.04805. — The contextual backbone; trained to predict text, not to make cosine space meaningful.
- Reimers & Gurevych (2019). "Sentence-BERT." EMNLP 2019, arXiv:1908.10084. — Canonical proof that "just embed it" fails unless the space is trained toward the intended similarity; qubric replaces SBERT's learned alignment with an instructed one — crucial where no labeled trace-similarity data exists.
- Ethayarajh (2019). "How Contextual are Contextualized Word Representations?" EMNLP 2019, arXiv:1909.00512. — Contextual spaces are anisotropic cones; the shared component swamps content differences — geometric background for authorship dominance.
- Li et al. (2020). "On the Sentence Embeddings from Pre-trained Language Models" (BERT-flow). EMNLP 2020, arXiv:2011.05864. — Frequency-biased anisotropy fixable by post-hoc transformation; same intervention family as consensus centering, different confound.
- Gao, Yao & Chen (2021). "SimCSE." EMNLP 2021, arXiv:2104.08821. — Alignment/uniformity is the modern vocabulary for our goal, but SimCSE hard-codes one generic notion of "same"; qubric makes the positive-set definition a per-query artifact.

#### A.5 What embeddings actually capture

- Adi et al. (2017). "Fine-grained Analysis of Sentence Embeddings…" ICLR 2017, arXiv:1608.04207; Conneau et al. (2018). "What you can cram into a single vector." ACL 2018, arXiv:1805.01070. — Founding probing methodology; single vectors entangle length, order, surface form — the paradigm behind our pillar battery, and early evidence that length is a strong embedded confound.
- Stamatatos (2009). "A Survey of Modern Authorship Attribution Methods." JASIST 60(3). — Authorship fingerprints live in low-level surface statistics (character n-grams, function words) exactly like those saturating raw traces.
- Rivera-Soto et al. (2021). "Learning Universal Authorship Representations" (LUAR). EMNLP 2021. — Authorship style is a strong, transferable signal that contrastive training readily latches onto; also a natural probe for residual authorship signal in our representations.
- Wegmann, Schraagen & Nguyen (2022). "Same Author or Just Same Topic?" RepL4NLP 2022, arXiv:2204.04907. — The exact dual of our problem (content-invariant style vs. authorship-invariant content); their conclusion that disentanglement requires explicit controls supports rubric specification plus centering.

#### A.6 Long-document representation

- Yang et al. (2016). "Hierarchical Attention Networks." NAACL 2016. — Documents as hierarchies; qubric's rubric decomposition is a semantic rather than positional hierarchy.
- Beltagy, Peters & Cohan (2020). "Longformer." arXiv:2004.05150. — The "scale the context window" answer; qubric takes the summarize-then-embed route instead.
- Cohan et al. (2020). "SPECTER." ACL 2020, arXiv:2004.07180. — Document similarity improves when training signal encodes the task-relevant relation (citations); precedent for defining similarity by an external relation.

### B. Modern embedders, rerankers, and LLM-mediated representation

#### B.1 General-purpose embedders and their benchmark

- Karpukhin et al. (2020). "Dense Passage Retrieval…" (DPR). EMNLP 2020, arXiv:2004.04906. — The bi-encoder + contrastive recipe all later embedders inherit; one fixed similarity function, no way to specify which aspects drive it.
- Izacard et al. (2021). "Unsupervised Dense Information Retrieval…" (Contriever). TMLR 2022, arXiv:2112.09118. — Similarity from within-document co-occurrence; on traces this surfaces style/authorship artifacts.
- Wang et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5). arXiv:2212.03533. — The "query:"/"passage:" prefix is a degenerate ancestor of prompt-conditioned similarity.
- Li et al. (2023). "Towards General Text Embeddings…" (GTE). arXiv:2308.03281. — Representative of the "one embedding fits all" position our thesis says fails on authorship-confounded traces.
- Xiao et al. (2023). "C-Pack" (BGE). SIGIR 2024, arXiv:2309.07597. — The off-the-shelf embedders our experiments show are authorship-dominated on raw traces.
- Nussbaum et al. (2024). "Nomic Embed." TMLR, arXiv:2402.01613. — 8K context with a fixed enum of task prefixes — coarse conditioning, not a programmable equivalence relation.
- Sturua et al. (2024). "jina-embeddings-v3." arXiv:2409.10173. — Task-specific LoRA adapters concede one geometry can't serve all similarity notions; the set of notions is still finite and baked in at training.
- Wang et al. (2024). "Improving Text Embeddings with Large Language Models" (E5-Mistral). ACL 2024, arXiv:2401.00368. — Instruction-prefixed LLM embedder; instructions condition the query side within trained task templates.
- Lee et al. (2024). "NV-Embed." ICLR 2025, arXiv:2405.17428; BehnamGhader et al. (2024). "LLM2Vec." COLM 2024, arXiv:2404.05961. — LLM-backbone embedders: capacity is abundant; our claim is it's misdirected without a similarity specification.
- Lee et al. (2024). "Gecko." arXiv:2403.20327; Lee et al. (2025). "Gemini Embedding." arXiv:2503.07891; OpenAI (2024). text-embedding-3 (announcement); Zhang, Li et al. (2025). "Qwen3 Embedding." arXiv:2506.05176. — The frontier commercial/open baselines; Qwen3 is the strongest open instruction-aware comparison point.
- Kusupati et al. (2022). "Matryoshka Representation Learning." NeurIPS 2022, arXiv:2205.13147. — Structures the vector by fidelity; qubric's concatenation is instead semantically indexed by rubric section.
- Muennighoff et al. (2022). "MTEB." EACL 2023, arXiv:2210.07316; Enevoldsen et al. (2025). "MMTEB." ICLR 2025, arXiv:2502.13595; RTEB (HF blog, 2025). — The leaderboard ecosystem; contains no task where the criterion of similarity varies per query; its own private-set response concedes overfitting.
- Weller et al. (2025). "On the Theoretical Limitations of Embedding-Based Retrieval" (LIMIT). arXiv:2508.21038. — Sign-rank proof that no single fixed embedding space represents all relevance relations; the theoretical backbone for per-rubric re-representation.

#### B.2 Instruction-conditioned and user-defined similarity (closest prior art)

- Su et al. (2022). "One Embedder, Any Task" (Instructor). Findings of ACL 2023, arXiv:2212.09741. — Founding promptable embedding; instructions are short trained-in task tags, conditioning is opaque, no invariance mechanism.
- Asai et al. (2022). "Task-aware Retrieval with Instructions" (TART). Findings of ACL 2023, arXiv:2211.09260. — First to frame relevance itself as instruction-dependent; retrieval-oriented, not a document-to-document similarity structure.
- Deshpande et al. (2023). "C-STS: Conditional Semantic Textual Similarity." EMNLP 2023, arXiv:2305.15093. — The most direct formalization of "similarity depends on a stated criterion"; SOTA embedders and GPT-4 do poorly. Qubric scales this from one-sentence conditions on sentence pairs to multi-section rubrics on 10k+-token traces.
- Tu et al. (2024). "Linguistically Conditioned Semantic Textual Similarity." ACL 2024, arXiv:2406.03673. — "Answer the condition, then compare answers" is structurally qubric's judge-extract-then-embed at sentence scale.
- Yoo et al. (2024). "Hyper-CL." ACL 2024, arXiv:2403.09490. — Hypernetwork maps a condition to a projection: one linear map per criterion; qubric's criterion is executed nonparametrically by an LLM, so unseen criteria need no training.
- CASE (2025/2026). "Condition-Aware Sentence Embeddings…" EACL 2026, arXiv:2503.17279. — Subtracting a condition-only component is kin to consensus centering; still sentence-level, single-condition.
- InBedder (2024). "Answer is All You Need." ACL 2024, arXiv:2402.09642. — Nearest published mechanism: represent a text by answers to an instruction-as-question. Differences: single instruction vs. structured per-instance rubric; short texts; hidden-state pooling vs. re-embedded extracted text; no invariance objective.
- Weller et al. (2024). "Promptriever." ICLR 2025, arXiv:2409.11136. — Free-form per-query relevance in a bi-encoder; query-time scoring rather than a reusable rubric-conditioned space over documents.
- Weller et al. (2024). "FollowIR." arXiv:2403.15246; Oh et al. (2024). "InstructIR." arXiv:2402.14334; Song et al. (2025). "IFIR." NAACL 2025, arXiv:2503.04644; Sun et al. (2024). "MAIR." EMNLP 2024, arXiv:2410.10127. — The benchmark literature documenting that trained-in instruction following is shallow, overfits instruction styles, and degrades with instruction complexity — motivating delegation of criterion execution to a general LLM at inference time.
- Muennighoff et al. (2024). "GRIT / GritLM." arXiv:2402.09906. — One model that both generates and embeds; could host qubric end-to-end, but embedding instructions remain task tags.
- Mysore, Cohan & Hope (2022). "Aspire." NAACL 2022, arXiv:2111.08366; Singh et al. (2022). "SciRepEval" (SPECTER2). EMNLP 2023, arXiv:2211.13308. — Aspect-conditional and format-conditional document representations; conditioning granularity is a small learned set, not open-ended rubric text.

#### B.3 Reranking and LLM-as-relevance-judge

- Nogueira & Cho (2019). "Passage Re-ranking with BERT." arXiv:1901.04085; Nogueira, Jiang & Lin (2020). "monoT5." Findings of EMNLP 2020, arXiv:2003.06713. — Accurate similarity needs joint reading of query and document; qubric gets joint reading via the judge, then amortizes it into vectors.
- Sun et al. (2023). "RankGPT." EMNLP 2023, arXiv:2304.09542 (also LRL, arXiv:2305.02156). — Frontier LLMs hold a superior latent relevance function accessible only pairwise/listwise at O(n·m) cost; qubric compiles that function, specialized by a rubric, into one pass per document.
- Pradeep et al. (2023). "RankVicuna" / "RankZephyr." arXiv:2309.15088 / 2312.02724. — Judge behavior is distillable into small open models — a route to cheapening the judge stage.
- Thomas et al. (2024). "LLMs can accurately predict searcher preferences." SIGIR 2024, arXiv:2309.10621; Upadhyay et al. (2024). "UMBRELA." arXiv:2406.06519. — Rubric-guided LLM assessment at industrial scale, one scalar per (query, doc); qubric generalizes the output from a grade to structured factual text that is then embedded.

#### B.4 LLM-generated text as representation

- Gao et al. (2022). "HyDE." ACL 2023, arXiv:2212.10496. — Foundational "embed the LLM-generated text instead"; qubric applies the factorization on the document side, under an explicit rubric.
- Wang, Yang & Wei (2023). "Query2doc." EMNLP 2023, arXiv:2303.07678; Zhuang et al. (2024). "PromptReps." EMNLP 2024, arXiv:2404.18424; Jiang et al. (2023). "PromptEOL." Findings of EMNLP 2024, arXiv:2307.16645; Springer et al. (2024). "Echo embeddings." arXiv:2402.15449. — The prompt-shapes-the-embedding tradition; prompts select a bottleneck, not a task-specific equivalence relation, and hidden-state artifacts inherit model-specific geometry.
- Benara et al. (2024). "QA-Emb." NeurIPS 2024, arXiv:2405.16714. — Text as a vector of LLM answers to natural-language questions; very close prior art. Binary answers vs. free-text sections; questions optimized for one regression target vs. per-instance rubrics; no invariance machinery.
- Sun et al. (2024). "CQG-MBQA." ICLR 2025, arXiv:2410.03435. — Automatic discriminative question generation at corpus scale — the nearest thing to an automated rubric generator; targets generic discriminativeness rather than a chosen equivalence relation.
- Ravfogel et al. (2023). "Description-Based Text Similarity." COLM 2024, arXiv:2305.12517. — States our motivating point: similarity is underspecified until the notion of sameness is; operationalizes one fixed notion, qubric makes the notion an input.
- Chen et al. (2023). "Dense X Retrieval." EMNLP 2024, arXiv:2312.06648. — LLM-extracted propositions as the indexing unit; qubric's extraction is proposition-ization steered by a rubric into fixed comparable sections.
- Opitz et al. (2025). "Interpretable Text Embeddings…: A Survey." EMNLP 2025, arXiv:2502.14862. — Maps the neighborhood; no surveyed method combines per-query criterion specification with cross-system invariance on long logs.

#### B.5 Long-document and long-context embedding

- Khattab & Zaharia (2020). "ColBERT." SIGIR 2020, arXiv:2004.12832; Santhanam et al. (2021). "ColBERTv2." NAACL 2022, arXiv:2112.01488. — Token-level multi-vector representations; on traces, token-level matching amplifies harness boilerplate.
- Chen et al. (2024). "BGE-M3." Findings of ACL 2024, arXiv:2402.03216. — Strongest open long-input hybrid baseline for raw-trace embedding.
- Zhu et al. (2024). "LongEmbed." EMNLP 2024, arXiv:2404.12096. — Documents the raw capability gap at 32k tokens; longer windows don't decide what drives similarity.
- Günther et al. (2024). "Late Chunking." arXiv:2409.04701. — Position-indexed chunks vs. qubric's rubric-section-indexed, semantically aligned "chunks" comparable across traces.

### C. Transformation, invariance, and disentanglement

#### C.1 Post-processing and isotropy

- Mu, Bhat & Viswanath (2018). "All-but-the-Top." ICLR 2018, arXiv:1702.01417. — Canonical "remove dominant common directions" post-processor; attribute-agnostic, same information source — the simplest geometric ancestor of consensus centering.
- Su et al. (2021). "Whitening Sentence Representations" (BERT-whitening). arXiv:2103.15316. — Full-covariance normalization; cannot distinguish authorship variance from content variance when the former dominates.
- Timkey & van Schijndel (2021). "All Bark and No Bite: Rogue Dimensions…" EMNLP 2021, arXiv:2109.04404. — A handful of high-variance nuisance dimensions can make a space effectively blind to the similarity of interest — the low-dimensional analogue of the fingerprint-reader finding.
- Conneau et al. (2018). "Word Translation Without Parallel Data" (CSLS). ICLR 2018, arXiv:1710.04087. — Centering against local consensus at the similarity level; a relative of per-task cross-system centering.

#### C.2 Concept erasure and its limits

- Elazar & Goldberg (2018). "Adversarial Removal of Demographic Attributes…" EMNLP 2018, arXiv:1808.06640. — Post-hoc classifiers recover "removed" attributes: the classic erasure-incompleteness result supporting the change-of-source argument.
- Gonen & Goldberg (2019). "Lipstick on a Pig." NAACL 2019, arXiv:1903.03862. — Nuisance information is distributed through the geometry, not confined to the removed direction.
- Ravfogel et al. (2020). "Null It Out" (INLP). ACL 2020, arXiv:2004.07667. — Standard post-hoc geometric erasure baseline; iterative projection consumes many dimensions when the nuisance is high-rank, damaging content.
- Ravfogel et al. (2022). "Linear Adversarial Concept Erasure" (R-LACE). ICML 2022, arXiv:2201.12091; "Adversarial Concept Erasure in Kernel Space." EMNLP 2022, arXiv:2201.12191. — Minimal linear surgery defeats linear adversaries only; protection against one nonlinear adversary does not transfer — you cannot enumerate who might read authorship out of the space.
- Belrose et al. (2023). "LEACE: Perfect Linear Concept Erasure in Closed Form." NeurIPS 2023, arXiv:2306.03819. — SOTA geometric alternative: provably perfect in the linear regime; residual nonlinear signal and content collateral remain.
- Ravfogel, Goldberg & Cotterell (2023). "Log-linear Guardedness and its Implications." ACL 2023, arXiv:2210.10012. — Even "perfect" linear erasure does not stop downstream leakage; invariance guarantees over the embedding space are the wrong abstraction.
- Fan et al. (2025). "The Medium Is Not the Message." arXiv:2507.01234. — LEACE-style erasure of source/medium confounds from document embeddings — the closest published analogue of our problem, solved geometrically; the natural head-to-head baseline.
- Akbari, Afshari & Boddeti (2026). "Obliviator…" arXiv:2603.07529. — 2026 SOTA nonlinear erasure: prior kernel methods leave 8–14% recoverability, and true nonlinear guardedness costs task utility — exactly the sufficiency/invariance tension we sidestep.

#### C.3 Fair / invariant representation learning

- Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks" (DANN). JMLR 17, arXiv:1505.07818. — The retrain-with-domain-labels alternative (agent ID as domain); needs labels and training, inherits adversarial-equilibrium leakage.
- Edwards & Storkey (2016). "Censoring Representations with an Adversary." ICLR 2016, arXiv:1511.05897; Louizos et al. (2016). "The Variational Fair Autoencoder." ICLR 2016, arXiv:1511.00830; Moyer et al. (2018). "Invariant Representations without Adversarial Training." NeurIPS 2018, arXiv:1805.09458. — The censoring/factorization/mutual-information family; all constrain an encoder rather than switching the information source.
- Zhao & Gordon (2019). "Inherent Tradeoffs in Learning Fair Representations." NeurIPS 2019, arXiv:1906.08386. — Proven lower bounds: invariance and utility trade off within a fixed representation; our claim is that changing the source moves the achievable frontier.

#### C.4 Style vs. content disentanglement

- Shen et al. (2017). "Style Transfer from Non-Parallel Text by Cross-Alignment." NeurIPS 2017, arXiv:1705.09655; John et al. (2019). "Disentangled Representation Learning for…Style Transfer." ACL 2019, arXiv:1808.04339. — The canonical shared-content/separated-style formulations; supervised, adversarial, in-space.
- Lample et al. (2019). "Multiple-Attribute Text Rewriting." ICLR 2019, arXiv:1811.00552. — Influential skeptical result: adversarially "disentangled" latents still leak attributes, and disentanglement isn't even necessary for control — parallels abandoning in-space surgery.
- Wieting & Gimpel (2018). "ParaNMT-50M." ACL 2018, arXiv:1711.05732. — Paraphrase-invariance as data-driven invariance to surface form; cross-agent traces of the same task are the agentic analogue of paraphrase pairs, exploited via consensus rather than training.
- Patel et al. (2025). "StyleDistance." NAACL 2025, arXiv:2410.12757. — Uses LLM generation to manufacture the invariance signal — a sibling of judge re-description, aimed at isolating style where we isolate content.

#### C.5 Authorship attribution and obfuscation

- Uchendu et al. (2020). "Authorship Attribution for Neural Text Generation." EMNLP 2020. — First systematic which-generator-wrote-this study; the seed of model fingerprinting.
- Uchendu, Le & Lee (2023). "Attribution and Obfuscation of Neural Text Authorship." SIGKDD Explorations 25(1), arXiv:2210.10488; Huang, Chen & Shu (2024). "Authorship Attribution in the Era of LLMs." SIGKDD Explorations, arXiv:2408.08946. — The umbrella surveys; notably, neither treats harness/scaffold-level attribution — evidence of the gap.
- Xing et al. (2024). "ALISON: Fast and Effective Stylometric Authorship Obfuscation." AAAI 2024, arXiv:2402.00835. — Text-space authorship removal (rewrite the artifact): the third alternative family between geometric erasure and re-description; judge extraction is an extreme content-anchored rewrite.

#### C.6 Model and agent fingerprinting

- Sun et al. (2025). "Idiosyncrasies in Large Language Models." ICML 2025, arXiv:2502.12150. — ~97% model attribution; idiosyncrasies survive paraphrase, translation, and summarization — why the judge output must still be consensus-centered rather than assumed fingerprint-free.
- Li et al. (2023). "Origin Tracing and Detecting of LLMs" (Sniffer). arXiv:2304.14072. — Early watermark-free model attribution from intrinsic output statistics.
- La Cava et al. (2025). "Authorship Attribution in Multilingual Machine-Generated Texts." arXiv:2508.01656. — Fingerprints persist across languages; attribution methods transfer poorly.
- Wu et al. (2026). "LLM DNA: Tracing Model Evolution via Functional Representations." ICLR 2026 (oral), arXiv:2509.24496. — Behavioral embeddings of models recover family/lineage phylogenies; raw trace embeddings do per-trace DNA extraction — exactly the axis we make invariant.
- Panickssery, Bowman & Feng (2024). "LLM Evaluators Recognize and Favor Their Own Generations." NeurIPS 2024, arXiv:2404.13076; Davidson et al. (2024). "Self-Recognition in Language Models." Findings of EMNLP 2024, arXiv:2407.06946. — Self-recognition causally drives self-preference (though introspective self-recognition is unreliable); caution for the judge stage and support for consensus centering.
- Guo et al. (2026). "Code Fingerprints: Disentangled Attribution of LLM-Generated Code." arXiv:2603.04212. — Attribution-side disentanglement of content vs. model style in generated code — the artifact type dominating SWE-bench traces; their benchmark could stress-test our invariance.
- Oderinwale (2026). "Agent Trajectories as Programs." arXiv:2606.16988. — Coding-agent traces attribute to scaffold+model at 85.7% (11% chance); deterministic scaffolds near-perfectly predictable. The perfect adversarial complement: it embraces the fingerprint we remove.
- Lugoloobi et al. (2026). "Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces." arXiv:2605.14786. — Model identification at 96% F1 from action traces alone; fingerprint signal lives in action structure, not just wording — text-level obfuscation alone cannot achieve invariance.
- Kang et al. (2026). "Whose Agent Are You?" arXiv:2606.20910. — Framework/harness-level attribution at 97% from multi-layer behavioral signatures; corroborates sibling-configuration fingerprinting from a security angle.

*Gap note: before 2026, machine-text attribution at the agent/harness level was essentially absent; all existing work is attacker-side. No prior work targets removing agent fingerprints from trace representations.*

### D. Efficient benchmarking, performance prediction, and statistical machinery

#### D.1 Benchmark compression / subset selection

- Maia Polo et al. (2024). "tinyBenchmarks." ICML 2024, arXiv:2402.14992. — ~100 curated examples estimate full-benchmark scores within ~2%; blends observed responses with IRT predictions — the canonical comparison for our probe-count/MAE trade-off, but correctness-only.
- Vivek et al. (2024). "Anchor Points." EACL 2024, arXiv:2309.08638. — Selects representative examples from correctness/confidence correlation clusters; direct ancestor of greedy probe selection, chosen from correctness rather than trace behavior.
- Kipnis et al. (2025). "metabench." ICLR 2025, arXiv:2407.12844. — Six leaderboard benchmarks compressed to <3% of items at ~1% RMSE; strongest evidence that score matrices are low-rank — the structural assumption our kernel regression exploits.
- Perlitz et al. (2024). "Efficient Benchmarking (of Language Models)." NAACL 2024, arXiv:2308.11696. — DIoR reliability metric; rank decisions flip under innocuous compressions — the vocabulary for arguing our estimates are decision-grade.
- Yauney et al. (2025/2026). "How Reliable is Language Model Micro-Benchmarking?" ICLR 2026 (oral), arXiv:2510.08730. — The resolution floor: at extreme budgets item-selection methods cannot separate models within ~3.5–4 points; our claim is that trace content breaks this floor because a probe carries far more than one bit.
- SparseEval (2026). arXiv:2602.07909; REP-CORE (2026). arXiv:2602.00710. — 2026 SOTA correctness-matrix compression; REP-CORE shows richer-than-correctness signals (hidden states) help but requires white-box access — our trace embeddings are black-box.
- Li et al. (2025). "ATLAS: Adaptive Testing for LLM Evaluation." arXiv:2511.04689; Truong et al. (2025). "Reliable and Efficient Amortized Model-based Evaluation." ICML 2025, arXiv:2503.13335. — CAT/IRT machinery; Truong et al. predict item difficulty from item content — the complementary direction of the same amortization idea (they embed items, we embed traces).

#### D.2 Performance prediction

- Ruan, Maddison & Hashimoto (2024). "Observational Scaling Laws." NeurIPS 2024, arXiv:2405.10938. — Models embedded by benchmark-score vectors; capabilities are smooth functions of a few latent dimensions — our reference-system geometry at the coarsest granularity.
- Owen (2024). "How predictable is language model benchmark performance?" arXiv:2401.04757; Ye et al. (2023). "How Predictable Are Large Language Model Capabilities?" Findings of EMNLP 2023, arXiv:2305.14947; Snell et al. (2024). "Predicting Emergent Capabilities by Finetuning." COLM 2025, arXiv:2411.16035. — Compute-side and record-matrix prediction, and cross-scale forecasting; delimits our claim (we predict a specific agent system, scaffold included, with compute unknown).
- Burnell et al. (2023). "Revealing the structure of language model capabilities." arXiv:2306.10062; Maimon et al. (2025). "From Benchmarks to Skills." arXiv:2507.20208. — Psychometric low-rank latent capability structure over model populations — the object DKPS estimates geometrically.
- Perlitz et al. (2024). "BenchBench." arXiv:2407.13696. — Benchmark-agreement conclusions hinge on reference-set choices; governs how we report cross-model correlation claims.

#### D.3 DKPS lineage (direct machinery)

- Duderstadt, Helm & Priebe (2023). "Comparing Foundation Models using Data Kernels." arXiv:2305.05126. — Origin of the data-kernel construction: models represented by embedding Gram matrices on probe data, compared benchmark-free with valid confidence sets.
- Helm, Duderstadt, Park & Priebe (2025). "Statistical inference on black-box generative models in the data kernel perspective space." ACL 2025, arXiv:2410.01106. — DKPS proper: model-level Euclidean representations from responses to a query collection, supporting population-level inference — the core citation for our machinery.
- Acharyya, Trosset, Priebe & Helm (2024). "Consistent estimation of generative model representations in the DKPS." arXiv:2409.17308; Acharyya, Agterberg, Park & Priebe (2025). "Finite-sample concentration for response-based embeddings." COMPSTAT 2026, arXiv:2511.08307. — Consistency and concentration theory warranting the few-probe regime; the variance side of our bias–variance blend.
- Helm, Duderstadt, Park & Priebe (2024). "Tracking the Perspectives of Interacting Language Models." EMNLP 2024, arXiv:2406.11938. — DKPS as a dynamic monitor over model populations.
- Helm, Johnson & Priebe (2026). "Query-efficient model evaluation using cached responses." ICML 2026, arXiv:2605.07096. — Closest prior work anywhere: predicts a new model's benchmark score from few probe responses vs. cached reference coverage in DKPS, ensembled with the sample score, >10× fewer queries on MATH/LegalBench/MedQA/WMT-14. Our project is its agentic-trace generalization (qubric representations, product kernels, fusion, greedy probe selection, leave-one-model-family-out).
- Browder et al. (2026). "DKPS Performance Guarantees for Synthetic Data…" arXiv:2602.05106; Helm, Liu & Yang (2026). "Jailbreak susceptibility prediction…via behavioral geometry." arXiv:2605.26409. — Precedents for attaching performance guarantees to DKPS coordinates, and independent evidence that few-probe DKPS predicts global behavioral properties (AUPRC 0.94 with ~98% fewer probes).
- Trosset & Priebe (2024). "Continuous Multidimensional Scaling." arXiv:2402.04436. — The classical-MDS backbone; supports out-of-sample insertion of a new agent into a fixed reference geometry — exactly our probe-time operation.

#### D.4 Evaluation methodology and leakage

- Sainz et al. (2023). "NLP Evaluation in trouble." Findings of EMNLP 2023, arXiv:2310.18018. — Contamination must be measured per benchmark; leakage can be between models, not just train/test.
- Singh et al. (2025). "The Leaderboard Illusion." arXiv:2504.20879. — Leaderboards as biased, expensive measurement instruments; our probe-based prediction is a partial remedy but must avoid reference-set biases.
- Goel et al. (2025). "Great Models Think Alike and this Undermines AI Oversight" (CAPA). ICML 2025, arXiv:2502.04313. — Chance-adjusted error-overlap similarity; judge scores inflated by judge–evaluatee similarity, and model errors increasingly correlated — the strongest published argument that naive cross-validation over agents overstates accuracy; also a correctness-only similarity baseline for our kernels.
- Liang, Garg & Zilouchian Moghaddam (2025). "The SWE-Bench Illusion." arXiv:2506.12286. — Memorization inflates Verified resolve rates; probe selection must be robust to memorization-inflated instances.

#### D.5 Agent benchmarking and its cost

- Jimenez et al. (2024). "SWE-bench." ICLR 2024 (oral), arXiv:2310.06770; OpenAI (2024). "Introducing SWE-bench Verified." — Our corpus's substrate and its 500-instance evaluation universe; even curating an agent benchmark took 93 annotators.
- Zhang et al. (2025). "SWE-bench-Live." NeurIPS 2025 D&B, arXiv:2505.23419; Deng et al. (2025). "SWE-Bench Pro." arXiv:2509.16941. — Contamination-resistant successors; natural transfer targets where cached full coverage is perpetually incomplete by design.
- Kapoor et al. (2024). "AI Agents That Matter." NeurIPS 2024, arXiv:2407.01502. — The cost-aware framing in which our method's value is measured; accuracy-only leaderboards incentivize costly scaffolds.
- Kapoor, Stroebl et al. (2025). "Holistic Agent Leaderboard (HAL)." ICLR 2026, arXiv:2510.11977. — ~$40K per holistic sweep; releases 2.5B tokens of agent traces — both the cost problem we attack and a corpus our method can consume.
- Ndzomga (2026). "Efficient Benchmarking of AI Agents." arXiv:2603.23749. — Nearest agent-specific subset selection; finds absolute-score prediction collapses under scaffold shift — precisely the gap trace-content DKPS regression closes.
- Song, Sutawika et al. (2026). "Pace: A Proxy for Agentic Capability Evaluation." arXiv:2607.02032. — Predicts agentic scores from cheap non-agentic proxy tasks on the underlying LLM; must marginalize away scaffold effects that our probe run of the actual agent system captures.
- Ge et al. (2026). "Agent Psychometrics." arXiv:2604.00594. — IRT extended to agentic coding with task features, LLM/scaffold ability decomposition, ~$22K full-run costs; the most direct contemporary competitor on SWE-bench Verified — uses task content and correctness, not trace content.

#### D.6 Routing and model embeddings from behavior

- Hu et al. (2024). "RouterBench." arXiv:2403.12031; Ong et al. (2024). "RouteLLM." ICLR 2025, arXiv:2406.18665. — The cached-outcome-matrix infrastructure pattern and per-query success prediction; same statistical object (model×instance success surface), different marginal.
- Zhuang et al. (2024). "EmbedLLM." ICLR 2025, arXiv:2410.02223; Song et al. (2025). "IRT-Router." ACL 2025, arXiv:2506.01048; Chen et al. (2025). "IrtNet." arXiv:2510.00844; Patel, Cocke & Joshi (2026). "LOCUS." arXiv:2601.21082. — Learned model embeddings from correctness records; LOCUS's embedding geometry recovers model families by clustering — the very structure that makes naive cross-validation leak and our leave-one-model-family-out protocol necessary.

### E. Agent trace analysis, trajectory evaluation, and LLM-as-judge

#### E.1 Trace analysis beyond pass/fail

- Mehtiyev & Assunção (2026). "Beyond Resolution Rates: Behavioral Drivers of Coding Agent Success and Failure." arXiv:2604.02547. — 9,374 leaderboard trajectories, 13-symbol action alphabet, paired-comparison designs; closest large-scale mining of our corpus, but symbolic statistics rather than content-faithful embedding, no invariance or reliability treatment.
- Shu et al. (2026). "What Resolve Rate Hides" (TraceProbe). arXiv:2607.06184. — Canonical action taxonomy with per-scaffold adapters; explicitly states raw trajectories are not comparable across scaffolds — the most direct documentation of format fragmentation motivating authorship invariance.
- Majgaonkar et al. (2025). "Understanding Code Agent Behaviour." arXiv:2511.00197. — Failed trajectories longer with higher variance; localization succeeds in 72–81% of failing runs — trace content carries signal orthogonal to pass/fail, and surface statistics are agent-confounded.
- Gupta et al. (2026). "Dissecting Model Behavior Through Agent Trajectories." arXiv:2606.17454. — 138K trajectories under one forced scaffold; the complementary comparability strategy (control the harness) to our post-hoc invariant representation.
- Martinez & Franch (2025). "Dissecting the SWE-Bench Leaderboards." arXiv:2506.17208. — Profiles 80 submissions; documents the heterogeneity of the ecosystem we embed.
- Cemri et al. (2025). "Why Do Multi-Agent LLM Systems Fail?" (MAST). NeurIPS 2025, arXiv:2503.13657. — 14-mode failure taxonomy with an LLM-annotator pipeline validated at κ≈0.88; fixed global taxonomy vs. our instance-specific rubrics.
- AgentDebug (2025). arXiv:2509.25370; Deshpande et al. (2025). "TRAIL." arXiv:2505.08638. — Step-level failure localization; TRAIL shows frontier long-context LLMs localize errors in traces at ~11% joint accuracy — grounding for our per-trace judge unreliability finding.
- Dong, Lu & Zhu (2024). "AgentOps." arXiv:2411.05285; Transluce (2025). "Docent." transluce.org/introducing-docent. — Trace observability infrastructure and interactive LLM-powered transcript analysis; Docent is the closest tooling analogue — query-time and interactive, where we produce fixed reusable vectors with quantified reliability.
- Podivilov et al. (2026). "AgentLens." arXiv:2607.06624. — Whole-trajectory review (instruction following, tool use, self-verification) with LLM-generated reviews; shares the trajectory-as-evaluand thesis, outputs free text not embeddings.

#### E.2 Process and trajectory-level evaluation

- Lightman et al. (2023). "Let's Verify Step by Step." ICLR 2024, arXiv:2305.20050. — Origin of process-over-outcome supervision; our rubric-sectioned descriptions instantiate it for software agents.
- AgentPRM (2025). WWW 2026, arXiv:2511.08325. — Step-wise promise/progress rewards for agent trajectories; scalar rewards for policy improvement vs. our factual descriptions for representation.
- Zhuge et al. (2024). "Agent-as-a-Judge." ICML 2025, arXiv:2410.10934. — Canonical trajectory-level judge (90%+ alignment at ~2% human cost); alignment is corpus-level — echoing our aggregation-rescues-agreement finding.
- Lù et al. (2025). "AgentRewardBench." arXiv:2504.08942. — 1,302 expert-annotated trajectories; no judge excels everywhere — the meta-evaluation validating that trajectory judges must themselves be measured.
- Xue et al. (2025). "An Illusion of Progress?" (WebJudge). arXiv:2504.01382; Ma et al. (2024). "AgentBoard." NeurIPS 2024, arXiv:2401.13178. — Trajectory judging and progress-rate metrics in other agent domains.
- Pan et al. (2024). "SWE-Gym." ICML 2025, arXiv:2412.21139. — Trajectory-level verifiers trained on SWE traces carry signal; also standardized trajectory data, in contrast to our fragmented leaderboard corpus.

#### E.3 LLM-as-judge foundations

- Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023, arXiv:2306.05685. — Establishes the methodology and the bias catalog (position, verbosity, self-enhancement).
- Liu et al. (2023). "G-Eval." EMNLP 2023, arXiv:2303.16634. — Criterion-decomposed judging; documents LLM-judge preference for LLM-written text.
- Wang et al. (2023). "Large Language Models Are Not Fair Evaluators." ACL 2024, arXiv:2305.17926. — Severe position bias; motivates extraction-then-embedding over direct pairwise trace comparison.
- Saito et al. (2023). "Verbosity Bias in Preference Labeling." arXiv:2310.10076. — Direct antecedent of our verbosity-as-mediator finding.
- Wataoka, Takahashi & Ri (2024). "Self-Preference Bias in LLM-as-a-Judge." arXiv:2410.21819. — Judges favor low-perplexity-under-own-distribution text even when not self-generated — predicts same-family leakage without exact self-authorship.
- Li et al. (2025). "Preference Leakage." arXiv:2502.01534. — Names and quantifies family leakage between generator and judge; strong support for our leave-one-model-out protocol.
- Ye et al. (2024). "Justice or Prejudice?" (CALM). ICLR 2025, arXiv:2410.02736. — 12-bias quantification framework; where our two trace-specific effects sit.
- Verga et al. (2024). "Replacing Judges with Juries" (PoLL). arXiv:2404.18796. — Panels of diverse judges beat single large judges at lower cost; supports multi-judge design and consensus centering.
- Bavaresco et al. (2024). "JUDGE-BENCH." ACL 2025, arXiv:2406.18403; "Rating Roulette" (2025). Findings of EMNLP 2025, arXiv:2510.27106; Thakur et al. (2024). "Judging the Judges." arXiv:2406.12624. — The judge-reliability literature (cross-task variability, seed inconsistency, kappa vs. percent agreement) whose methodology our 12–28% trace-level and 0.82–0.95 aggregated agreement figures extend into the long-trace regime.

#### E.4 Rubric- and checklist-based evaluation (closest prior art for qubric's rubric stage)

- Arora, Wei et al. (2025). "HealthBench." arXiv:2505.08775. — 48,562 physician-written conversation-specific rubric criteria applied by a model grader, with grader meta-evaluation — the flagship of instance-specific rubric evaluation; outputs scores, not representations.
- Cook et al. (2024). "TICK." arXiv:2410.03608; Lee et al. (2024). "CheckEval." EMNLP 2025, arXiv:2403.18771. — LLM-generated instance-specific checklists measurably raise judge–human and inter-judge agreement — the mechanism our query-specific rubric relies on.
- Viswanathan et al. (2025). "Checklists Are Better Than Reward Models" (RLCF). NeurIPS 2025, arXiv:2507.18624; Gunjal et al. (2025). "Rubrics as Rewards." arXiv:2507.17746. — Instance-adaptive criteria beat monolithic scorers as training signal; we use the same insight for representation.
- Wang & Blanco (2026). "Generating and Refining Dynamic Evaluation Rubrics." arXiv:2605.30568; Dhole & Agichtein (2026). "RubricRAG." arXiv:2603.20882; Rao & Callison-Burch (2026). "Autorubric." arXiv:2603.00077; Pan et al. (2026). "RubricEval." arXiv:2603.25133. — The 2026 wave on automatic rubric generation, its quality controls, its standardization, and its meta-evaluation (even GPT-4o ~56% on hard subsets; explicit reasoning reduces inter-judge variance).
- Chen et al. (2026). "From Holistic Evaluation to Structured Criteria" (survey). arXiv:2606.08625. — The survey to cite for situating qubric in the rubric landscape; contains no rubric-to-embedding pipeline — marking our gap.

#### E.5 Structured compression of long logs; LLM annotator reliability

- Wang et al. (2023). "Recursively Summarizing…Long-Term Dialogue Memory." arXiv:2308.15022; Ou & Lapata (2025). "Context-Aware Hierarchical Merging." Findings of ACL 2025, arXiv:2502.00977; (2024) "Two-Stage Summarization for Long Dialogues." arXiv:2410.06520. — The segment-then-extract architecture and its faithfulness risks — the same content-fidelity problem our judge stage manages.
- Xiao et al. (2025). "AgentDiet: Reducing Cost of LLM Agents with Trajectory Reduction." FSE 2026, arXiv:2509.23586. — 40–60% of coding-agent trace tokens removable without performance loss — independent evidence that traces are massively semantically compressible.
- Gilardi, Alizadeh & Kubli (2023). "ChatGPT Outperforms Crowd Workers." PNAS 120(30), arXiv:2303.15056. — Anchor of the LLM-as-annotator reliability literature; our judge-agreement analysis is its methodology applied to 10K-token traces.

#### E.6 SWE-bench ecosystem

- Yang et al. (2024). "SWE-agent." NeurIPS 2024, arXiv:2405.15793; Wang et al. (2024). "OpenHands." ICLR 2025, arXiv:2407.16741. — The two dominant scaffolds/trace formats: major "authors" in our corpus whose idioms are exactly the authorship signal we remove.
- Yang et al. (2025). "SWE-smith." arXiv:2504.21798; Badertdinov et al. (2025). "SWE-rebench." NeurIPS 2025, arXiv:2505.20411. — Trajectory data at scale in standardized formats, and decontaminated continuous evaluation.
- Yu et al. (2025). "UTBoost." ACL 2025. — 345 mislabeled patches affecting 24.4% of Verified leaderboard entries: ground-truth label noise in the leaderboard we mine — strengthening content-based representations over pass/fail labels.
- Zhu et al. (2025). "Establishing Best Practices for Building Rigorous Agentic Benchmarks (ABC)." NeurIPS 2025, arXiv:2507.02825. — Task/outcome-validity flaws misestimate performance up to ~100% relative; supports our reliability-first framing.

#### E.7 Embedding/clustering of behaviors, dialogues, reasoning traces

- Liu et al. (2022). "dial2vec." EMNLP 2022, arXiv:2210.15332. — Whole-dialogue embeddings; short two-party interactions, no tool logs, length extremes, or authorship confounds.
- Ge et al. (2025). "Learning Informative Trajectory Embeddings." AAMAS 2025, arXiv:2501.09327. — RL trajectory embeddings encode agent "style" and skill — evidence naive trace embeddings encode authorship.
- Zhao et al. (2023). "ExpeL." AAAI 2024, arXiv:2308.10144; Chung et al. (2026). "InsightEmb." arXiv:2608.04761; "Retrieval-of-Thought" (2025). arXiv:2509.21743; Zheng et al. (2026). "Trajectory Graphs for Pre-Execution Error Diagnosis." arXiv:2607.27443. — Trajectory representation for the agent's own reuse (retrieval, thought graphs, GNN risk flags); we embed for external analysis with invariance and reliability guarantees.

*End of bibliography. Total: ~150 verified entries across five strands.*
