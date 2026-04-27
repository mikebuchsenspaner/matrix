# Matrix — Robô de Investimentos Quantitativo

Sistema de trading algorítmico baseado em Machine Learning para classificação binária de oportunidades de compra.

---

## Estrutura do projeto

```
Projeto investiment/
├── data/
│   └── market_data.csv       ← CSV com OHLCV histórico
├── models/
│   └── trade_model.pkl       ← modelo treinado (gerado por train.py)
├── backtest_results/
│   ├── threshold_summary.csv
│   ├── backtest_best.csv
│   ├── trades_best.csv
│   └── summary_best.txt
├── reports/
│   ├── walkforward_folds.csv
│   ├── walkforward_summary.csv
│   └── walkforward_predictions.csv
│
├── settings.py               ← TODOS os parâmetros do projeto
├── features.py               ← feature engineering + labeling
├── simulation.py             ← lógica de simulação (compartilhada)
├── train.py                  ← treina e salva o modelo
├── backtest.py               ← backtest na janela de teste
├── walkforward.py            ← validação walk-forward
├── main.py                   ← inferência (decisão atual)
├── get_data.py               ← download de dados reais (yfinance)
├── generate_data.py          ← geração de dados sintéticos (testes)
└── requirements.txt
```

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Passo a passo para rodar

### 1. Dados

**Opção A — Dados reais (recomendado)**
```bash
python get_data.py
```
Baixa 5 anos de dados diários do AAPL via yfinance (~1250 linhas).
Edite `ticker` dentro do script para mudar o ativo.

**Opção B — Dados sintéticos (para testes rápidos)**
```bash
python generate_data.py --rows 2000
```

### 2. Treinar o modelo

```bash
python train.py
```

Saída esperada:
- Distribuição do label (proporção de 0s e 1s)
- Acurácia, ROC-AUC, Average Precision no conjunto de teste
- Distribuição de probabilidades previstas
- Top 15 features por importância
- Modelo salvo em `models/trade_model.pkl`

### 3. Backtest

```bash
python backtest.py
```

Testa a estratégia nos dados de teste (últimos 20% do período).
Compara múltiplos thresholds e salva o melhor.

### 4. Walk-Forward

```bash
python walkforward.py
```

Validação mais robusta: retreina o modelo em múltiplas janelas temporais
e avalia a consistência dos resultados entre folds.

### 5. Inferência (decisão atual)

```bash
python main.py
```

Usa o modelo treinado para emitir `BUY` ou `NO BUY` com base nos dados mais recentes.

```bash
python main.py --threshold 0.45
python main.py --csv data/outro_ativo.csv --threshold 0.40
```

---

## Parâmetros principais (`settings.py`)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `HORIZON` | 5 | Barras à frente para definir o label |
| `MIN_RETURN` | 0.003 | Retorno mínimo para label = 1 (0.3%) |
| `TEST_SIZE` | 0.20 | Fração do dataset para teste |
| `THRESHOLDS` | [0.30..0.50] | Thresholds testados no backtest |
| `USE_REGIME_FILTER` | False | Filtro de regime (True = menos trades) |
| `STOP_LOSS` | 0.02 | Stop loss de 2% |
| `TAKE_PROFIT` | 0.04 | Take profit de 4% |
| `MAX_HOLD_BARS` | 5 | Máximo de 5 barras em posição |
| `FEE_RATE` | 0.001 | Taxa de 0.1% por entrada/saída |
| `WF_INITIAL_TRAIN_SIZE` | 0.60 | Treino inicial no walk-forward (60%) |
| `WF_TEST_SIZE` | 0.10 | Janela de teste por fold (10%) |

---

## Como interpretar os resultados

### Backtest (`backtest_results/summary_best.txt`)

| Métrica | O que significa | Bom sinal |
|---|---|---|
| Alpha vs B&H | Retorno da estratégia − buy and hold | Positivo |
| Max drawdown | Maior queda do pico até o vale | < 15% |
| Sharpe | Retorno ajustado ao risco | > 0.5 |
| Profit factor | Lucros totais / perdas totais | > 1.0 |
| Win rate | % de trades lucrativos | > 50% |
| Total trades | Número de operações | ≥ 10 para ser válido |

### Walk-Forward (`reports/walkforward_summary.csv`)

O walk-forward é mais confiável que o backtest simples porque:
- Retreina o modelo em cada janela (simula o mundo real)
- Avalia em dados nunca vistos durante aquele treino
- Mostra se a estratégia é consistente ao longo do tempo

**O que olhar:**
- `folds_positive` — em quantos folds o retorno foi positivo? (ex: 6/8 é bom)
- `avg_sharpe` — Sharpe médio entre folds
- `avg_alpha_vs_buy_hold` — alpha médio
- `avg_trades` — se < 3 trades por fold, o resultado não é estatisticamente confiável

### Poucos trades?

Se o backtest mostrar < 5 trades:
1. Verifique `USE_REGIME_FILTER = False` no `settings.py`
2. Reduza `MIN_RETURN` (ex: de 0.003 para 0.002)
3. Use dados reais com mais histórico (`python get_data.py`)
4. Diminua o threshold testado (ex: adicione 0.25 em `THRESHOLDS`)

---

## Ordem de execução

```
get_data.py (ou generate_data.py)
         ↓
      train.py
         ↓
     backtest.py
         ↓
   walkforward.py
         ↓
       main.py
```

---

## Aviso

Este projeto é uma **prova de conceito** para fins educacionais.
Não deve ser usado para operar dinheiro real sem validação adicional,
gestão de risco profissional e análise de overfitting em dados out-of-sample.
