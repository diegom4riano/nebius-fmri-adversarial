The Nebius Serverless AI Builders Challenge gave me the opportunity to test a hypothesis I'd been sitting on.

Clinical AI models, brain connectivity networks, ECG classifiers, physiological time series, are routinely evaluated for adversarial robustness using tools built for image classifiers. A model that passes such an evaluation is deployed with a robustness certificate. If the certificate is wrong, the vulnerability is invisible, until someone exploits it to flip a diagnosis.

The core idea is geometric. Imagine you are at the top of a mountain with a thick forest and you want to find the best way down. But you can't see beyond the trees. You have only the feel of the slope under your feet to go on. You always go step downhill. That's  gradient attack. It works on regular terrain, but when the mountain is full of razor-thin ridges next to deep valleys, following the slope becomes a trap, you zig-zag on the ridges. A second-order method adds curvature awareness: it knows when the terrain is twisting, and corrects for it.

On a cardiac ECG rhythm classifier (well-conditioned surface, such as a ball): both methods converge. The gradient attacks are enough, the terrain is regular. On a brain fMRI connectivity model for neurological patterns detection (GNN + RNN, ill-conditioned surface):
→ AutoAttack (current gold standard): 17.9% attack success
→ KAPPA (second-order): 60.7%

3.4× gap means that the robustness benchmarks based on gradient attacks might overestimating safety in any clinical model based on GNNs or RNNs that are not fully normalized. That is a a significant part of what is being deployed in medical AI today.

It was genuinely fun to work through this. Thanks to Nebius for the challenge and for the H200 that made it possible in a few hours.

Full write-up on Medium → [link in comments]

#NebiusServerlessChallenge #MedicalAI #AdversarialML
