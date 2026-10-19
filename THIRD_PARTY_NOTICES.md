# Attribution and redistribution

`baselines/vendor/model/` contains code from the HGCLR authors' repository
`https://github.com/wzh9969/contrastive-htc`, author commit recorded by the
verified protocol: `322a7ff2d83c878534bed25bb288cf4479d00363`.
Its MIT license is preserved in `baselines/vendor/LICENSE`.
`baselines/hgclr_verified/vendor/` is the separately frozen dependency snapshot
used by the verified retraining source; its license is also preserved.
Local byte hashes in `provenance/source_manifest.json` identify the actual
files exported here. A commit reference alone does not certify all local files
as unmodified upstream code.

The four methods are research comparators, with ESAW adaptations and limits
documented in `docs/EXPERIMENT_STATUS.md`. HTC-CLIP is a paper reconstruction,
not an official released implementation. No third-party model weights or
third-party benchmark data are included. Obtain public pretrained models from
their owners under the applicable terms.

Original ESAW-Link contributions and accompanying software documentation are
released under the root MIT license. The upstream HGCLR MIT copyright and
license notices above remain applicable and are preserved separately. The root
license does not replace third-party copyright notices or grant rights to
external model weights, private institutional data or articles.

The project implementations of HiSR and HyILR are in
`baselines/esaw_htc/recent_models.py`. Reference author repositories used in the
scientific fidelity audit are:

- HiSR: https://github.com/zjcjason/HiSR, reference commit
  `a678f7df7087dab8ca4fc41f70f0afe0dfe2bd74`.
- HyILR: https://github.com/havelhakimi/HyILR, reference commit
  `2dcffd074d9736457083ac7a3541869c3878de28`.

No checkout or verbatim source snapshot from either author repository is
distributed here. GitHub did not report a repository license for these two
reference repositories when checked on 19 October 2026; this release does not
purport to license those repositories or authorize copying their source.
The reference publications are HiSR, COLING 2025
(https://aclanthology.org/2025.coling-main.378/), and HyILR, ACL SRW 2025
(https://aclanthology.org/2025.acl-srw.63/).

`baselines/esaw_htc/soft_verbalizer.py` is a small project compatibility
implementation of the OpenPrompt SoftVerbalizer interface, retained with the
historical source package; it is not used by the four reported HTC rows.
External Python packages are installed separately under their own licenses.

