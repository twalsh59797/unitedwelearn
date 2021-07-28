Hierarchical Sigmoid and Hierarchical Cross-entropy losses for Object Detection with COCO are implemented in this repository, extending Detectron2.


## Detectron

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md),
or the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License~~~~

Detectron2 is released under the [Apache 2.0 license](LICENSE).


# Running with Hierarchical Loss

To train and evaluate with hierarchical loss, add the following configurations to training: 
```
MODEL.RETINANET.HIERARCHICAL_FOCAL_LOSS True
TEST.EVAL_HIERARCHICAL True
```

To configure the hierarchial levels and respective weights use, e.g::

```
MODEL.RETINANET.HIERARCHICAL_LOSS_WEIGHTS 0.7,0.1,0.1,0.1
MODEL.RETINANET.MAX_HIERARCHY_LEVELS 4
```