# Detecção de fraude em cartão de crédito

Solução para a competição [Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11) (Kaggle Playground Series, Season 6, Episode 11).

O objetivo é prever a probabilidade de cada transação ser fraude (`is_fraud`). A métrica é a área sob a curva ROC (ROC AUC).

## Dados

| Conjunto | Linhas | Observação |
|---|---|---|
| `train.csv` | 699.960 | Cerca de 1,67% de fraudes |
| `test.csv` | 299.984 | Sem a coluna alvo |
| [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) | 20.000 | Opcional, licença CC0 |

São 24 atributos sobre a transação, o cliente, o dispositivo e sinais de risco. Os dados da competição são sintéticos e **não estão incluídos neste repositório**: baixe-os na página da competição.

## Abordagem

1. Checagem de qualidade e análise exploratória.
2. Regressão logística como modelo de referência.
3. Engenharia de atributos, mantida apenas se aumentar a AUC do LightGBM.
4. LightGBM, XGBoost e CatBoost com validação cruzada estratificada em 5 partes.
5. Combinação dos quatro modelos por média de ranks, com pesos escolhidos por seleção gulosa.
6. Interpretação e geração do arquivo de submissão.

Os modelos de boosting usam taxa de aprendizado 0,1, até 2.000 árvores e parada antecipada após 100 rodadas sem melhora. Quando presente, o dataset original entra somente no treino desses três modelos, nunca na validação.

## Estrutura

```text
.
├── README.md
├── fraude_s6e11.ipynb
└── data/                 # só para execução local
    ├── train.csv
    ├── test.csv
    └── credit_card_fraud_2026.csv   # opcional
```

## Como executar

### No Kaggle (recomendado)

1. Entre na competição e aceite as regras.
2. Crie um notebook e importe `fraude_s6e11.ipynb` em **File > Import Notebook**.
3. Confira em **Input** se os dados da competição estão anexados.
4. Clique em **Save Version > Save & Run All**.
5. Na aba **Output** da versão salva, envie o `submission.csv`.

### No computador

Requer Python 3. Coloque os arquivos na pasta `data/` e instale as dependências:

```bash
pip install numpy pandas matplotlib scipy scikit-learn lightgbm xgboost catboost jupyter
```

Execute no Linux ou macOS:

```bash
DATA_DIR=data jupyter nbconvert --to notebook --execute fraude_s6e11.ipynb \
  --output executado.ipynb --ExecutePreprocessor.timeout=-1
```

No Windows (PowerShell):

```powershell
$env:DATA_DIR = "data"
jupyter nbconvert --to notebook --execute fraude_s6e11.ipynb --output executado.ipynb --ExecutePreprocessor.timeout=-1
```

## Arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `submission.csv` | `id` e probabilidade `is_fraud` de cada transação de teste |
| `resultados.csv` | AUC, desvio entre folds, PR AUC e tempo de cada modelo |

## Resultados

_A preencher após a execução com os dados da competição._

| Modelo | AUC (OOF) | PR AUC |
|---|---|---|
| Regressão logística | | |
| LightGBM (atributos originais) | | |
| LightGBM (com engenharia) | | |
| XGBoost | | |
| CatBoost | | |
| Blend por ranks | | |

**Placar público do Kaggle:** _a preencher_

## Limitações

- A parada antecipada e a escolha dos pesos usam os dados de validação, então a AUC OOF é um pouco otimista.
- Os dados são sintéticos, e os resultados não valem automaticamente para transações reais.
- Próximos passos: codificação do alvo dentro de cada fold, várias sementes e busca de hiperparâmetros.

## Autores

- Cauê
- Rafael
