# Detecção de Fraude em Cartão de Crédito (Kaggle S6E11)

Solução desenvolvida para a competição **[Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11)** (*Kaggle Playground Series — Season 6, Episode 11*), como parte da atividade prática de resolução de desafios em plataformas de *crowdsourcing* de Ciência de Dados.

## Visão Geral do Problema

- **Objetivo:** Prever a probabilidade (`is_fraud` entre `0` e `1`) de cada transação financeira no conjunto de teste ser fraudulenta.
- **Métrica de Avaliação:** Área sob a Curva ROC (**ROC AUC**), que avalia a capacidade do modelo de atribuir scores mais altos às transações fraudulentas em comparação às legítimas.
- **Desafio Principal:** Forte desbalanceamento entre as classes — apenas **1,67%** das transações na base de treino são fraudes.

---

## Conjunto de Dados

Os dados da competição são sintéticos e **não estão incluídos neste repositório** (devem ser baixados diretamente na página da competição no Kaggle).

| Arquivo | Registros | Colunas | Descrição |
| :--- | :---: | :---: | :--- |
| `train.csv` | 699.960 | 26 | Base de treino com `id`, 24 atributos e o alvo `is_fraud` (1,67% positivas) |
| `test.csv` | 299.984 | 25 | Base de teste para submissão (sem a coluna `is_fraud`) |
| `credit_card_fraud_2026.csv` | 20.000 | 26 | Dataset externo opcional ([uditjain13](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026), licença CC0) |

### Dicionário dos 24 Atributos Originais

- **Dados da Transação:** `amount_usd` (valor em dólar), `time_of_day_hour` (hora do dia), `day_of_week` (dia da semana), `is_foreign_transaction` (compra internacional).
- **Comportamento e Histórico:** `hours_since_last_txn` (horas desde a última compra), `txn_count_last_24h` (qtd. de compras nas últimas 24h), `velocity_score` (score de velocidade de uso), `prior_disputes` (contestações anteriores).
- **Cliente, Cartão e Lojista:** `customer_age` (idade do cliente), `card_age_months` (tempo do cartão em meses), `account_balance_usd` (saldo da conta), `merchant_category` (categoria do lojista), `merchant_risk_score` (risco do lojista), `is_new_merchant` (primeira compra nesse lojista), `card_type` (tipo do cartão).
- **Autenticação, Dispositivo e Alertas:** `auth_method` (método de autenticação), `channel` (canal da compra), `device_type` (dispositivo), `cvv_retry_count` (tentativas erradas de CVV), `distance_from_home_km` (distância de casa), `used_vpn` (uso de VPN), `ip_country_mismatch` (país do IP diferente), `billing_shipping_mismatch` (endereço de cobrança diferente da entrega), `is_ai_generated_scam_attempt` (tentativa de golpe gerado por IA).

---

## Metodologia e Pipeline Técnico

O fluxo desenvolvido no notebook `fraude.ipynb` segue 6 etapas principais:

### 1. Checagem de Integridade e Análise Exploratória (EDA)
- Verificação de qualidade: **0 valores ausentes** (no treino e no teste) e **0 linhas duplicadas**.
- Padronização das 6 variáveis booleanas para inteiros (`0` e `1`).
- Cálculo da **AUC univariada** para medir o poder de separação isolado de cada variável numérica:
  1. `merchant_risk_score`: **0,74894**
  2. `velocity_score`: **0,71079**
  3. `cvv_retry_count`: **0,70808**
  4. `is_new_merchant`: **0,64224**
  5. `customer_age`: **0,62472**
  6. `txn_count_last_24h`: **0,62031**

### 2. Engenharia de Atributos (24 → 40 variáveis)
Foram criados **16 novos atributos** baseados em hipóteses de comportamento fraudulento, sem utilizar a variável alvo `is_fraud` (prevenindo *data leakage*):

| Atributo Criado | Fórmula / Lógica | Hipótese de Negócio |
| :--- | :--- | :--- |
| `red_flags` | Soma das 6 flags booleanas + indicador de erro de CVV | Vários sinais de alerta simultâneos aumentam muito o risco |
| `amount_to_balance` | `amount_usd / (abs(account_balance_usd) + 1)` | Fraudadores tendem a tentar gastar quase todo o saldo disponível |
| `log_amount` | `log1p(amount_usd)` | Reduz a assimetria de compras de valor muito alto |
| `log_distance` | `log1p(distance_from_home_km)` | Suaviza distâncias extremas em relação à residência do titular |
| `risk_x_velocity` | `merchant_risk_score * velocity_score` | Interação entre lojista arriscado e alta frequência de transações |
| `txn_per_gap` | `txn_count_last_24h / (hours_since_last_txn + 0.1)` | Identifica rajadas de compras em intervalos muito curtos |
| `hour_sin`, `hour_cos` | Seno e cosseno da hora (`2π * hora / 24`) | Preserva a natureza cíclica do relógio (23h próximo de 0h) |
| `is_night` | `1` se `time_of_day_hour < 6`, senão `0` | Captura compras realizadas durante a madrugada |
| `auth_channel` | Combinação `auth_method \| channel` | Perfis específicos de canal + autenticação têm riscos distintos |
| `merchant_auth` | Combinação `merchant_category \| auth_method` | Certas categorias de loja exigem autenticação mais rigorosa |
| `*_freq` (5 colunas) | Frequência relativa de ocorrência na base | Valores raros ou muito repetidos ajudam a diferenciar padrões |

### 3. Estratégia de Validação e Modelo de Referência (Baseline)
- **Validação Cruzada:** `StratifiedKFold` com **3 folds** (`seed = 42`), garantindo exatamente a mesma proporção de 1,67% de fraudes em cada partição. Todos os modelos usam exatamente os mesmos folds.
- **Baseline:** Pipeline com `StandardScaler` (numéricas), `OneHotEncoder` (categóricas) e **Regressão Logística** (`max_iter = 2000`) treinado apenas nos 24 atributos originais.

### 4. Modelos de Gradient Boosting
Três algoritmos baseados em árvores de decisão foram treinados sobre os 40 atributos com parada antecipada (*early stopping* de 50 rodadas):
- **LightGBM:** `learning_rate=0.03`, `num_leaves=7`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=5.0`.
- **XGBoost:** `learning_rate=0.03`, `max_depth=3`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.7`, `tree_method="hist"`.
- **CatBoost:** `learning_rate=0.1`, `depth=4`, `l2_leaf_reg=5.0`, suporte nativo às variáveis categóricas.

### 5. Combinação de Modelos (Rank Blend)
- Como a métrica ROC AUC avalia apenas a ordenação das probabilidades, as previsões fora da amostra (*Out-of-Fold — OOF*) de cada modelo foram convertidas em **postos (`ranks`)**.
- Um algoritmo de busca gulosa (*Hill Climbing* em 25 passos) encontrou os pesos ótimos para maximizar a AUC: **56% LightGBM + 44% CatBoost**.

---

## Resultados Obtidos

### Comparativo de Desempenho (Validação Cruzada 3-Fold OOF)

| Posição | Modelo | ROC AUC (OOF) | Desvio entre Folds | PR AUC | Tempo (min) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1º** | **Blend por Ranks (56% LGBM + 44% CatBoost)** | **0,94616** | **0,00224** | **0,50849** | 0,01* |
| 2º | LightGBM | 0,94571 | 0,00231 | 0,50469 | 1,79 |
| 3º | CatBoost | 0,94547 | 0,00217 | 0,50566 | 5,69 |
| 4º | XGBoost | 0,94538 | 0,00226 | 0,50249 | 3,31 |
| 5º | Regressão Logística (Referência) | 0,94172 | 0,00271 | 0,48898 | 0,22 |

*\*Tempo referente apenas à otimização dos pesos do blend sobre as previsões já calculadas.*

**Placar Público no Kaggle:** _preencher após upload do `submission.csv`_

### Impacto Operacional (Simulação de Triagem de Risco)

Ordenando as transações pela probabilidade prevista pelo *Blend*, avaliou-se quantas fraudes uma equipe de segurança barraria ao revisar manualmente apenas o topo mais arriscado:

| % de Transações Revisadas | % de Fraudes Capturadas | Precisão (Acerto nas Revisões) |
| :---: | :---: | :---: |
| Top **0,5%** | **24,7%** | **82,7%** |
| Top **1,0%** | **38,8%** | **64,7%** |
| Top **2,0%** | **53,7%** | **44,9%** |
| Top **5,0%** | **70,9%** | **23,7%** |

---

## Estrutura do Projeto e Execução

```text
.
├── README.md             # Documentação completa do projeto
├── fraude.ipynb          # Notebook com análise, modelagem e geração dos CSVs
├── resultados.csv        # Tabela comparativa de métricas dos modelos
├── submission.csv        # Arquivo final com 299.984 previsões para o Kaggle
└── dados/                # Pasta local com os datasets (não versionada)
    ├── train.csv
    └── test.csv
