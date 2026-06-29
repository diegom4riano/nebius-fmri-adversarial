The Nebius Serverless AI Builders Challenge gave me the opportunity I needed to test a hypothesis I'd been sitting on.

Clinical AI models — brain connectivity networks, ECG classifiers, physiological time series — are routinely evaluated for adversarial robustness using tools built for image classifiers. A model that passes that evaluation gets deployed with a robustness certificate. If the certificate is wrong, the vulnerability is invisible — until someone exploits it to flip a diagnosis.

The core idea is geometric. Imagine you're at the summit of a densely forested mountain and need to find the best path down — but you can't see beyond the trees. The only tool you have is feeling the slope under your feet. You always step downhill. That's a gradient attack. It works on regular terrain, but when the mountain has razor-thin ridges next to deep valleys, following the slope becomes a trap — you zig-zag along the ridges. A second-order method adds curvature awareness: it knows when the terrain is twisting, and corrects for it.

On a cardiac ECG rhythm classifier (well-conditioned surface, like a ball): both methods converge. Gradient attacks are sufficient — the terrain is regular. On a brain fMRI connectivity model used to detect neurological patterns (GNN + RNN, ill-conditioned surface):
→ AutoAttack (current gold standard): 17.9% attack success
→ KAPPA (second-order): 60.7%

3.4× gap — which means robustness benchmarks built on gradient attacks may be systematically overestimating safety for any clinical model based on GNNs or RNNs without full normalization. That covers a significant share of what's being deployed in medical AI today.

It was genuinely fun to work through this. Thanks Nebius for the challenge — and for the H200 that made it possible in a few hours.

Full write-up on Medium → [link in comments]

#NebiusServerlessChallenge #MedicalAI #AdversarialML
