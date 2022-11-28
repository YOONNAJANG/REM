
### FoCus modeling codes

evaluate_test.py 파일에 factual consistency 확인을 위한 메트릭인 FactCC 메트릭 추가.
해당 모델은 [factcc pretrained model](https://github.com/salesforce/factCC) 에서 다운로드 후 사용. 
test셋에서 question rewriting inference를 위한 부분 추가, **regen_question** argument가 True로 되어있으면 question rewriting 진행.


[dae-factuality](https://github.com/tagoyal/dae-factuality) 추가
하위 폴더에 model/ 디렉토리 안에 dae_w_syn_hallu 폴더가 있어야 함(13번 yoonna/focus_modeling/dae_factuality안에 있음). stanford-corenlp-4.4.0 도 설치되어 있어야 함.

evaluation은 13번 아래 경로에 있는 파일들로 진행 (이전 캐시 지우고 실행)

```
/home/mnt/yoonna/focus_modeling/our_data/test_ours.json
/home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json
```


윤나 추가 내용 부분

- train.sh 파일에 총 BART와 T5 각각 8가지 모델 (엑셀 참조 부탁!)

- classification_modules, generate_modules 파일을 합쳐서 cusgen_generate으로 바꿈 