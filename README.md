[Post-hoc Utterance Refining Method by Entity Mining for Faithful Knowledge Grounded Conversations](https://aclanthology.org/2023.emnlp-main.295.pdf)
===
Built by [Yoonna Jang](https://github.com/YOONNAJANG), [Suhyune Son](https://github.com/sonsuhyune), [Jeongwoo Lee](https://github.com/jeongwoolee-jason), [Junyoung Son](https://github.com/rgop13)

***
### Official codes for the paper: <br/>**[Post-hoc Utterance Refining Method by Entity Mining for Faithful Knowledge Grounded Conversations](https://aclanthology.org/2023.emnlp-main.295.pdf)**, acceㅎpted @[EMNLP 2023](https://aclanthology.org/volumes/2023.emnlp-main/).
***

Despite the striking advances in recent language generation performance, model-generated responses have suffered from the chronic problem of hallucinations that are either untrue or unfaithful to a given source. Especially in the task of knowledge grounded conversation, the models are required to generate informative responses, but hallucinated utterances lead to miscommunication. In particular, entity-level hallucination that causes critical misinformation and undesirable conversation is one of the major concerns. To address this issue, we propose a post-hoc refinement method called **REM**. It aims to enhance the quality and faithfulness of hallucinated utterances by refining them based on the source knowledge. If the generated utterance has a low source-faithfulness score with the given knowledge, REM mines the key entities in the knowledge and implicitly uses them for refining the utterances. We verify that our method reduces entity hallucination in the utterance. Also, we show the adaptability and efficacy of REM with extensive experiments and generative results.

<p align="center"><img src="rem_ex.png" width="380px" height="460px" title="REM Example"></img></p>



### Setting Environment
We trained the models under the setting of `python==3.7` and `torch==1.9.0`, with one RTX8000 GPU. 
Thanks to open source libraries, such as [transformers](https://github.com/huggingface/transformers), [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning), [wandb](https://github.com/wandb/wandb) we built our code on their codes. We also use [DAE](https://github.com/tagoyal/dae-factuality?tab=readme-ov-file) and [Distinct-N](https://github.com/neural-dialogue-metrics/Distinct-N) metrics, and we thank the authors for releasing the codes.


1.Make a virtual environment
    
    $conda create -n ENV_NAME python=3.8

2.Install `pytorch==1.9.0` according to your CUDA version.

    $conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

3.Install the required libraries.
    
    $pip install -r requirements.txt

4.Download DAE (dae_w_syn_hallu) [model checkpoint](https://drive.google.com/drive/folders/16NEL8T-JvhJPy7miVUbMELVE8ZOTYGit). As DAE relies on Stanford CoreNLP, the code below should be run in stanford-corenlp folder (Please refer [DAE](https://github.com/tagoyal/dae-factuality?tab=readme-ov-file) for help):
    
    $ nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &

5.Download data


    📦REM
    ┣ 📂data
    ┃ ┗ 📜train.json
    ┃ ┗ 📜valid.json
    ┣ 📂metrics
    ┃ ┗ 📜distincN.py
    ┃ ┗ 📂model
    ┃   ┗ 📂dae_w_syn_hallu
    ┣ 📂src
    ┣ 📜README.md
    ┗ 📜requirements.txt


### Training models
Uncomment the command lines in the **`train.sh`** file, to start training the model. 

    $ sh train.sh 


### Evaluation
Uncomment the command lines in the **`test.sh`** file, to evaluate the model on the test set. 

    $ sh test.sh


### Inference
Uncomment the command lines in the **`inference.sh`** file, to generate utterances with the trained models.

    $ sh inference.sh




### Citation
To use our data or source code, please cite our paper:

    @inproceedings{jang2023post,
      title={Post-hoc Utterance Refining Method by Entity Mining for Faithful Knowledge Grounded Conversations},
      author={Jang, Yoonna and Son, Suhyune and Lee, Jeongwoo and Son, Junyoung and Hur, Yuna and Lim, Jungwoo and Moon, Hyeonseok and Yang, Kisu and Lim, Heui-Seok},
      booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      pages={4844--4861},
      year={2023}
    }
