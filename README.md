# DRRN Model Variants (Hash input, Inverse dynamics) on Text Games 

Code for NAACL 2021 paper [Reading and Acting while Blindfolded: The Need for Semantics in Text Game Agents](https://arxiv.org/abs/2103.13552).

## Getting Started

- Install dependencies:
```bash
pip install jericho fasttext
```
- Run baseline DRRN:
```python
python train.py
```

- Run DRRN (hash):
```python
python train.py --hash_rep 1
```

- Run DRRN (inv-dy):
```python
python train.py --w_inv 1 --w_act 1 --r_for 1
```

Use ``--seed`` to specify game random seed. ``-1`` means episode-varying seeds (stochastic game mode), otherwise game mode is deterministic.

Zork I is played by default. More games are [here](https://github.com/princeton-nlp/calm-textgame/tree/master/games) and use ``--rom_path`` to specify which game to play.

## Citation
```
@inproceedings{yao2021blindfolded,
    title={Reading and Acting while Blindfolded: The Need for Semantics in Text Game Agents},
    author={Yao, Shunyu and Narasimhan, Karthik and Hausknecht, Matthew},
    booktitle={North American Association for Computational Linguistics (NAACL)},
    year={2021}
}
```
## Acknowledgements
The code borrows from [TDQN](https://github.com/microsoft/tdqn). 

For any questions please contact Shunyu Yao `<shunyuyao.cs@gmail.com>`.
