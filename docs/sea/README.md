# SEA 코드·실험 안내

현재 기본 구현은 **SEA (fix123)** 입니다. `python -m craftax.sea.train`은
fresh streaming discovery를 사용합니다. fix123는 세 수정(완전한 episode
contrast, fresh prediction streaming, 기준 상대 loss 가중치)을 반영한 실험명입니다.
**ICLR-SEA의 정확한 reproduction이라는 뜻은 아닙니다.**

- [원본과 구현 차이](alignment.md)
- [실행과 코드 구조](workflow.md)
- [기존 상세 실행 설명](../../SEA_README.md)

기존 데이터, 체크포인트, source snapshot은 보존합니다. 정리 과정에서 학습 예산,
네트워크, loss 가중치, clipping을 바꾸지 않습니다. 이전 구현은 각 run의 `source/`가
권위 있는 기록이며 현재 코드로 과거 실험을 설명하지 않습니다.
