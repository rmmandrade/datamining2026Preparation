# Correlações entre variáveis (secção 1.3.4)

Este documento explica a análise de correlações feita na secção **1.3.4 Feature Correlations** do notebook `notebook/02_data_understanding.ipynb`. Serve para o grupo perceber a análise e para reaproveitar no relatório. Todos os valores vêm dos resultados do notebook.

## Índice

1. [Objetivo](#1-objetivo)
2. [Porquê Spearman](#2-porquê-spearman)
3. [Passo 1: dados preparados para a correlação](#3-passo-1-dados-preparados-para-a-correlação-1341)
4. [Passo 2: heatmap com as variáveis novas](#4-passo-2-heatmap-com-as-variáveis-novas-1342)
5. [Passo 3: variáveis redundantes](#5-passo-3-variáveis-redundantes-1343)
6. [Passo 4: o efeito da antiguidade](#6-passo-4-o-efeito-da-antiguidade-1344)
7. [Passo 5: variáveis categóricas](#7-passo-5-variáveis-categóricas-1345)
8. [Conclusões para o clustering](#8-conclusões-para-o-clustering-1346)
9. [Limitações](#9-limitações)
10. [Perguntas prováveis na discussão](#10-perguntas-prováveis-na-discussão)

---

## 1. Objetivo

Antes de fazer clustering, é preciso saber **que variáveis dizem a mesma coisa** e **que relações são reais e quais são ilusórias**. Isto importa por duas razões:

- **Peso duplicado.** Algoritmos baseados em distâncias, como o K-Means, tratam cada variável como uma dimensão. Se entrarem três variáveis que medem o mesmo, esse tema conta três vezes e domina a formação dos clusters.
- **Relações enganadoras.** Duas variáveis podem parecer relacionadas só porque dependem de uma terceira. Se isso não for detetado, podemos tirar conclusões erradas sobre os segmentos.

A secção 1.3.2.4 já tinha um heatmap de Spearman sobre os dados brutos. A secção 1.3.4 completa essa análise em quatro pontos:

1. Volta a calcular as correlações com os valores inválidos tratados e inclui as variáveis criadas na 1.3.3.
2. Identifica os pares de variáveis redundantes.
3. Controla o efeito da antiguidade (tenure), algo que o notebook deixava para depois (*"we will check this by controlling for tenure in the future"*).
4. Mede as associações das variáveis categóricas, que o Spearman não consegue medir.

## 2. Porquê Spearman

| | Pearson | Spearman |
|---|---|---|
| O que mede | Relações em linha reta | Relações monótonas: quando uma sobe, a outra tende a subir (ou a descer) |
| Usa | Os valores | A ordem (rank) dos valores |
| Sensível a outliers | Muito | Pouco |
| Pressupõe distribuição normal | Para os testes, sim | Não |

Os nossos dados têm **distribuições muito assimétricas**, **muitos zeros** (add-ons, horas, valor) e **outliers extremos** (um `lifetime_value` de 3,4 milhões, salas de reunião com milhares de horas). Com Pearson, poucos valores extremos podiam criar ou esconder uma correlação. O Spearman só depende da ordem, por isso é mais adequado.

**Como ler o coeficiente ρ:** vai de −1 a +1.

- **ρ perto de +1:** as duas variáveis sobem juntas.
- **ρ perto de −1:** quando uma sobe, a outra desce.
- **ρ perto de 0:** não há relação monótona.

Uma regra prática: |ρ| < 0,3 é fraca, 0,3 a 0,7 é moderada e > 0,7 é forte.

---

## 3. Passo 1: dados preparados para a correlação (1.3.4.1)

### O problema
O heatmap da 1.3.2.4 usava os dados brutos, que ainda têm códigos que representam valores em falta e valores impossíveis, identificados na análise univariada:

- `999` em `days_since_last_visit`;
- `-1` em `n_colleagues_on_account`;
- idades fora do intervalo realista;
- `plan_days_per_week` igual a 0 ou 9;
- visitas, horas, passes, renovações e valores negativos.

Estes códigos entram no cálculo como se fossem valores reais. Um 999 em dias desde a última visita parece um cliente que não aparece há quase 3 anos.

### O que foi feito
A função `prepare_correlation_data` cria uma **cópia** dos dados, usada só nesta análise:

| Regra | Variável |
|---|---|
| Remover linhas totalmente duplicadas | Todas (15 linhas) |
| Idade fora de 16 a 100 passa a "em falta" | `age` |
| Fora de 1 a 7 dias passa a "em falta" | `plan_days_per_week` |
| Código 999 e valores negativos passam a "em falta" | `days_since_last_visit` |
| Código −1 passa a "em falta" | `n_colleagues_on_account` |
| Valores negativos passam a "em falta" | `total_visits`, `lifetime_value`, `n_renewals`, `meeting_room_hours`, `event_space_hours`, `guest_passes_issued` |

Depois, a função chama a `engineer_features` da 1.3.3, para as variáveis novas serem calculadas já sobre os dados corrigidos. O `df_raw` original **não é alterado**.

Os valores inválidos ficam como "em falta" e não são imputados. Cada correlação usa todas as linhas em que as duas variáveis do par são válidas (*pairwise deletion*). Assim não se perde uma linha inteira por causa de um único valor inválido, e a decisão de imputar fica para a fase de preparação dos dados.

### Valores em falta depois da preparação

| Variável | Valores em falta | Origem principal |
|---|---|---|
| `days_since_last_visit`, `recency_ratio` | 1.360 | Código 999 |
| `meeting_room_hours` | 1.311 | Valores negativos |
| `guest_passes_issued` | 1.276 | Valores negativos |
| `lifetime_value`, `revenue_per_visit`, `value_per_member` | 1.230 | Valores em falta originais e negativos |
| `age` | 700 | Fora de 16 a 100 |
| `n_colleagues_on_account` | 565 | Código −1 |
| `plan_days_per_week`, `visit_intensity_gap` | 97 | Valores 0 ou 9 |
| `n_renewals` | 66 | Negativos |
| `total_visits` | 42 | Negativos |
| `event_space_hours` | 8 | Negativos |

O dataset final tem 15.450 linhas e nenhuma variável perde mais de 9% dos valores.

### Variáveis analisadas
Entram 32 variáveis, entre numéricas, booleanas e as criadas na 1.3.3. Ficaram de fora:

- **As versões logarítmicas** (`*_log`), `membership_duration_months` e `total_account_members`. São transformações monótonas de outras variáveis (log, divisão por 30,44, soma de 1), por isso a ordem dos valores é igual e o Spearman daria exatamente os mesmos resultados.
- **As datas**, que já foram analisadas nas secções 1.3.2.5 e 1.3.2.6.

### Impacto da preparação
Comparando com o heatmap da 1.3.2.4:

| Par | ρ dados brutos | ρ dados preparados | Diferença |
|---|---|---|---|
| `days_since_last_visit` ↔ `lifetime_value` | −0,37 | −0,18 | +0,19 |
| `is_team_account` ↔ `n_colleagues_on_account` | 0,89 | 0,98 | +0,10 |
| `days_since_last_visit` ↔ `avg_weekly_visits` | −0,66 | −0,73 | −0,08 |
| `days_since_last_visit` ↔ `actual_visits_last_period` | −0,61 | −0,69 | −0,07 |
| `days_since_last_visit` ↔ `total_visits` | −0,51 | −0,57 | −0,06 |

**Interpretação:**
- Quase todas as correlações mudam menos de 0,1, por isso as conclusões da 1.3.2.4 continuam válidas no essencial.
- A grande exceção é **dias desde a última visita ↔ valor do cliente**: metade da correlação original era criada pelo código 999. Como a secção 1.4.2 refere, o 999 coincide quase sempre com `lifetime_value` igual a zero, o que criava uma relação artificial.
- As relações entre recência e visitas ficam **mais fortes** depois da limpeza, porque o código 999 estava a misturar ruído.
- `is_team_account` e `n_colleagues_on_account` ficam quase idênticas (0,98) depois de retirar o −1.

---

## 4. Passo 2: heatmap com as variáveis novas (1.3.4.2)

O heatmap está **agrupado por clustering hierárquico** (`sns.clustermap`). As variáveis são reordenadas para as parecidas ficarem lado a lado e formarem blocos visíveis.

- A distância entre duas variáveis é **1 − |ρ|**. Usamos o valor absoluto para que variáveis com correlação negativa forte (por exemplo, recência e visitas) também fiquem juntas, porque também dão a mesma informação.
- O método de ligação é `average` (média das distâncias entre grupos).

### Os cinco grupos

| Grupo | O que mede | Variáveis | ρ dentro do grupo |
|---|---|---|---|
| **Tamanho do plano** | A que é que o plano dá direito | `plan_tier_encoded`, `plan_days_per_week`, `entitled_visits_last_period` | 0,84 a 0,96 |
| **Atividade atual** | Quanto o membro usa o espaço | `avg_weekly_visits`, `actual_visits_last_period`, `utilization_rate_last_period`, `visit_intensity_gap` | 0,59 a 0,93 |
| **Antiguidade e valor acumulado** | Há quanto tempo é cliente e quanto acumulou | `membership_duration_days`, `total_visits`, `lifetime_value`, `value_per_member`, `n_renewals` | 0,44 a 0,74 |
| **Add-ons** | Uso de serviços extra | `uses_*`, `meeting_room_hours`, `event_space_hours`, `guest_passes_issued`, `addon_diversity_score` | 0,26 a 0,98 |
| **Sem relação** | — | `age`, `uses_bike_locker`, `is_team_account`, `n_colleagues_on_account` | \|ρ\| ≤ 0,03 com as restantes |

**Notas sobre os grupos:**
- A **atividade atual** está negativamente ligada a `days_since_last_visit` (−0,62 a −0,73): quem vai mais vezes esteve lá há menos tempo, o que é esperado.
- No grupo da **antiguidade**, o `recency_ratio` funciona como espelho negativo (−0,48 a −0,84).
- Nos **add-ons**, cada flag é quase igual à respetiva quantidade (0,96 a 0,98). O `addon_diversity_score` relaciona-se com todos os add-ons (0,26 a 0,63), sobretudo com as salas de reunião (0,60 a 0,63) e a impressão (0,51).
- No grupo **sem relação**, a única exceção é `value_per_member` com as variáveis de equipa (−0,20 e −0,22). Isto acontece por construção: o valor é dividido pelo tamanho da conta.

### O que dizem as variáveis criadas na 1.3.3

| Variável | Relações principais | Interpretação |
|---|---|---|
| `revenue_per_visit` | Utilização −0,62, `days_since_last_visit` +0,62, visitas semanais −0,55 | Um valor alto **não** indica cliente premium. Indica quem **paga e não aparece**: o valor é dividido por poucas visitas. |
| `value_per_member` | `lifetime_value` 0,97 | Quase igual a `lifetime_value`. 72% das contas são individuais, e para essas as duas variáveis são iguais. |
| `utilization_rate_last_period` | `visit_intensity_gap` 0,87, `actual_visits` 0,84, `avg_weekly_visits` 0,84 | Resume bem a atividade e já é relativa ao plano (−0,15 com `plan_tier`). |
| `visit_intensity_gap` | Utilização 0,87, `plan_days_per_week` −0,48 | Muito sobreposta à utilização. Fica negativa nos planos grandes, porque é mais difícil usar todos os dias a que se tem direito. |
| `recency_ratio` | `total_visits` −0,84, `days_since_last_visit` 0,81, tenure −0,69 | Mistura recência com antiguidade (ver passo 4). |
| `addon_diversity_score` | Salas de reunião 0,63, impressão 0,51, `total_visits` 0,47 | Bom resumo do uso de serviços extra, porque se relaciona com todos os add-ons. |
| `plan_tier_encoded` | `plan_days_per_week` 0,96, `entitled_visits` 0,84 | O plano define os dias de acesso. |

### Outras observações
- **`uses_day_passes`** é o único add-on negativamente ligado ao tamanho do plano, às renovações e à antiguidade (−0,12 a −0,15). Sugere utilizadores ocasionais, com planos pequenos.
- **Um ponto estranho para o grupo verificar:** `is_active` está negativamente ligado às visitas no último período (−0,32) e positivamente aos dias desde a última visita (0,17). Ou seja, os membros ativos visitaram **menos** no último período do que os que saíram, o que vai contra o esperado. Pode estar relacionado com as inconsistências de datas da secção 1.4.3 e deve ser confirmado quando as datas forem limpas.

---

## 5. Passo 3: variáveis redundantes (1.3.4.3)

A função `top_correlated_pairs` lista todos os pares com **|ρ| ≥ 0,8**. São 15:

| # | Variável 1 | Variável 2 | ρ |
|---|---|---|---|
| 1 | `is_team_account` | `n_colleagues_on_account` | 0,98 |
| 2 | `event_space_hours` | `uses_event_space` | 0,98 |
| 3 | `lifetime_value` | `value_per_member` | 0,97 |
| 4 | `guest_passes_issued` | `uses_guest_passes` | 0,96 |
| 5 | `meeting_room_hours` | `uses_meeting_rooms` | 0,96 |
| 6 | `plan_tier_encoded` | `plan_days_per_week` | 0,96 |
| 7 | `avg_weekly_visits` | `actual_visits_last_period` | 0,93 |
| 8 | `plan_days_per_week` | `entitled_visits_last_period` | 0,89 |
| 9 | `utilization_rate_last_period` | `visit_intensity_gap` | 0,87 |
| 10 | `actual_visits_last_period` | `utilization_rate_last_period` | 0,84 |
| 11 | `recency_ratio` | `total_visits` | −0,84 |
| 12 | `avg_weekly_visits` | `utilization_rate_last_period` | 0,84 |
| 13 | `plan_tier_encoded` | `entitled_visits_last_period` | 0,84 |
| 14 | `total_visits` | `actual_visits_last_period` | 0,81 |
| 15 | `days_since_last_visit` | `recency_ratio` | 0,81 |

### Sugestão: uma variável por grupo

| Grupo | Variáveis correlacionadas | Sugestão | Justificação |
|---|---|---|---|
| Conta de equipa | `is_team_account`, `n_colleagues_on_account` | `n_colleagues_on_account` | 0 colegas corresponde **exatamente** às contas individuais (confirmado no cruzamento das duas variáveis), por isso a flag não acrescenta nada. |
| Flag vs quantidade | `uses_event_space` / `event_space_hours`, `uses_guest_passes` / `guest_passes_issued`, `uses_meeting_rooms` / `meeting_room_hours` | A quantidade | Quantidade 0 já quer dizer "não usou", e a quantidade diz também *quanto* usou. |
| Valor | `lifetime_value`, `value_per_member` | `lifetime_value` | São diferentes só nas contas de equipa (28%). |
| Tamanho do plano | `plan_tier_encoded`, `plan_days_per_week`, `entitled_visits_last_period` | `plan_tier_encoded` ou `plan_days_per_week` | `entitled_visits_last_period` depende mais da antiguidade (0,69, contra 0,43 a 0,48). |
| Atividade atual | `avg_weekly_visits`, `actual_visits_last_period`, `utilization_rate_last_period`, `visit_intensity_gap` | Uma variável de intensidade | Se o plano também entrar no modelo, `utilization_rate_last_period` evita contar o tamanho do plano duas vezes, porque já é relativa ao plano. |
| Recência | `recency_ratio`, `days_since_last_visit`, `total_visits` | `days_since_last_visit` + antiguidade em separado | O `recency_ratio` mistura duas coisas diferentes (ver passo 4). |
| Visitas acumuladas | `total_visits`, `actual_visits_last_period` | Evitar `total_visits` em bruto | É sobretudo efeito da antiguidade (ver passo 4). |

O limite de 0,8 é uma **regra prática**. A escolha final depende das perspetivas de segmentação e faz-se na fase de clustering.

---

## 6. Passo 4: o efeito da antiguidade (1.3.4.4)

### O problema
Várias variáveis **acumulam com o tempo**: visitas totais, valor total, renovações e as flags `uses_*`, que querem dizer "alguma vez usou". Um cliente com 3 anos tem forçosamente mais de tudo do que um cliente com 3 meses. Por isso, estas variáveis podem parecer relacionadas entre si **só porque todas crescem com a antiguidade**, sem haver nenhum comportamento em comum.

A isto chama-se uma **variável de confusão** (*confounder*): uma terceira variável que influencia as outras duas e cria uma correlação entre elas.

### A solução: correlação parcial
A **correlação parcial** mede a relação entre duas variáveis depois de retirar o efeito de uma terceira. Responde à pergunta: *entre membros com a mesma antiguidade, a relação mantém-se?*

Para duas variáveis *x* e *y*, controlando a antiguidade *z*:

$$\rho_{xy \cdot z} = \frac{\rho_{xy} - \rho_{xz}\,\rho_{yz}}{\sqrt{(1-\rho_{xz}^2)(1-\rho_{yz}^2)}}$$

Intuitivamente, o numerador retira a parte da relação que é explicada por ambas estarem ligadas à antiguidade, e o denominador volta a pôr o resultado na escala de −1 a 1. A antiguidade é medida por `membership_duration_days` (de `membership_start` até `membership_end`, ou até à data de extração, 2024-01-01, para os membros ativos).

### Que variáveis dependem da antiguidade

| Variável | ρ com a antiguidade |
|---|---|
| `total_visits` | 0,74 |
| `lifetime_value` | 0,70 |
| `recency_ratio` | −0,69 |
| `entitled_visits_last_period` | 0,69 |
| `value_per_member` | 0,66 |
| `n_renewals` | 0,58 |
| `plan_days_per_week` | 0,48 |
| `plan_tier_encoded` | 0,43 |
| `addon_diversity_score` | 0,38 |
| `avg_weekly_visits` | 0,22 |
| `utilization_rate_last_period` | 0,11 |
| `revenue_per_visit` | −0,02 |
| `age`, `is_team_account`, `n_colleagues_on_account` | ≈ 0 |

As **variáveis acumuladas** (totais) dependem muito da antiguidade. As **taxas** (médias por semana, utilização) dependem pouco.

### Relações que enfraquecem

| Par | ρ normal | ρ parcial | Variação |
|---|---|---|---|
| `total_visits` ↔ `entitled_visits_last_period` | 0,58 | 0,14 | −0,44 |
| `total_visits` ↔ `lifetime_value` | 0,61 | 0,19 | −0,42 |
| `entitled_visits_last_period` ↔ `lifetime_value` | 0,59 | 0,21 | −0,38 |
| `total_visits` ↔ `n_renewals` | 0,49 | 0,11 | −0,38 |
| `recency_ratio` ↔ `lifetime_value` | −0,60 | −0,24 | −0,37 |
| `n_renewals` ↔ `lifetime_value` | 0,46 | 0,10 | −0,36 |
| `recency_ratio` ↔ `n_renewals` | −0,48 | −0,13 | −0,34 |
| `entitled_visits_last_period` ↔ `n_renewals` | 0,49 | 0,16 | −0,33 |
| `plan_days_per_week` ↔ `total_visits` | 0,40 | 0,08 | −0,32 |
| `plan_days_per_week` ↔ `lifetime_value` | 0,45 | 0,18 | −0,26 |

**Interpretação:** a maior parte do que ligava valor, visitas e renovações na secção 1.3.2.4 era **tempo como membro**, não comportamento. Um cliente com mais visitas totais não vale mais por ir mais vezes; vale mais porque está há mais tempo.

### Relações que ficam mais fortes

| Par | ρ normal | ρ parcial | Variação |
|---|---|---|---|
| `total_visits` ↔ `visit_intensity_gap` | 0,38 | 0,60 | +0,21 |
| `is_active` ↔ `total_visits` | −0,17 | −0,38 | +0,21 |
| `recency_ratio` ↔ `visit_intensity_gap` | −0,44 | −0,63 | +0,19 |
| `total_visits` ↔ `utilization_rate_last_period` | 0,59 | 0,77 | +0,18 |
| `lifetime_value` ↔ `revenue_per_visit` | 0,39 | 0,56 | +0,17 |
| `days_since_last_visit` ↔ `recency_ratio` | 0,81 | 0,96 | +0,16 |

**Interpretação:**
- Entre membros com a mesma antiguidade, quem tem mais visitas totais é quem **usa mais o plano** (0,77). Aqui a relação é de comportamento.
- `days_since_last_visit` e `recency_ratio` passam a ser quase iguais (0,96). Quando a antiguidade é conhecida, o `recency_ratio` não acrescenta informação.

### Implicação para o clustering
- **Não usar variáveis acumuladas em bruto** (`total_visits`, `lifetime_value`, `n_renewals`), sobretudo se a antiguidade também entrar no modelo.
- **Preferir taxas por unidade de tempo.** Para a segmentação por valor, uma variável como **valor por mês de membro** (`lifetime_value / membership_duration_months`) mede a contribuição económica sem a misturar com a antiguidade.

---

## 7. Passo 5: variáveis categóricas (1.3.4.5)

O Spearman precisa de valores ordenáveis, por isso não funciona com variáveis nominais como `gender`, `country` ou `industry`. Usámos duas medidas, ambas entre **0 (sem associação)** e **1 (associação perfeita)**.

### Cramér's V (categórica vs categórica)
Baseia-se no teste do qui-quadrado: compara a tabela cruzada observada com a que se esperaria se as variáveis fossem independentes. Usámos a **versão com correção de viés** (Bergsma, 2013), porque a versão simples sobrestima a associação quando as variáveis têm muitas categorias.

O p-value do qui-quadrado **não foi usado**. Com 15 mil linhas, quase qualquer diferença mínima dá "significativa", por isso o que interessa é o **tamanho** da associação, que é o que o V mede.

| | gender | country | industry | plan_tier | is_team_account | is_active |
|---|---|---|---|---|---|---|
| **gender** | — | 0,00 | 0,00 | 0,00 | 0,01 | 0,00 |
| **country** | 0,00 | — | 0,03 | 0,10 | 0,00 | 0,08 |
| **industry** | 0,00 | 0,03 | — | 0,12 | 0,01 | 0,12 |
| **plan_tier** | 0,00 | 0,10 | 0,12 | — | 0,01 | 0,32 |
| **is_team_account** | 0,01 | 0,00 | 0,01 | 0,01 | — | 0,01 |
| **is_active** | 0,00 | 0,08 | 0,12 | 0,32 | 0,01 | — |

### Correlation ratio η (categórica vs numérica)
Mede **que parte da variação de uma variável numérica é explicada pelas categorias**. Se as médias dos grupos forem muito diferentes entre si e os valores dentro de cada grupo forem parecidos, η é alto.

$$\eta = \sqrt{\frac{\sum_{g} n_g (\bar{y}_g - \bar{y})^2}{\sum_{i} (y_i - \bar{y})^2}}$$

Calculámos η sobre os **ranks** da variável numérica, para ser coerente com o Spearman e robusto aos outliers.

Valores principais:

| Categórica | Variáveis numéricas com maior η |
|---|---|
| `plan_tier` | `plan_days_per_week` 1,00 · `entitled_visits_last_period` 0,88 · `membership_duration_days` 0,52 · `visit_intensity_gap` 0,51 · `n_renewals` 0,49 · `lifetime_value` 0,47 · `value_per_member` 0,45 · `total_visits` 0,42 |
| `industry` | `plan_days_per_week` 0,22 · `entitled_visits_last_period` 0,21 · `membership_duration_days` 0,18 · `visit_intensity_gap` 0,17 · `n_renewals` 0,17 |
| `country` | `plan_days_per_week` 0,18 · `membership_duration_days` 0,18 · `entitled_visits_last_period` 0,18 · `addon_diversity_score` 0,18 · `n_renewals` 0,17 |
| `gender` | Todos ≤ 0,02 |

### Interpretação
- **`gender`** não tem relação com nada. Não ajuda a separar segmentos, só a descrevê-los.
- **`country` e `industry`** são quase independentes entre si (V = 0,03) e têm relações fracas com as variáveis numéricas (η ≤ 0,22), sobretudo com o tamanho do plano e a antiguidade. Servem para **perfilar** os clusters.
- **`plan_tier`** é a variável categórica mais informativa:
  - praticamente define `plan_days_per_week` (η = 1,00) e `entitled_visits_last_period` (0,88);
  - está moderadamente ligada à antiguidade, às renovações e ao valor (0,47 a 0,52);
  - é a categórica mais associada a ainda estar ativo (V = 0,32), o que confirma a maior retenção dos planos grandes vista na secção 1.5.
- **`is_team_account`** não está associada a nenhuma outra categórica (V ≤ 0,01).

---

## 8. Conclusões para o clustering (1.3.4.6)

1. **Qualidade dos dados.** Os códigos falsos e os valores impossíveis mudam pouco a maioria das correlações, mas distorcem as relações de recência. Têm de ser tratados antes do clustering.
2. **Redundância.** As 32 variáveis analisadas formam poucos grupos muito correlacionados: conta de equipa, flags e quantidades dos add-ons, valor, tamanho do plano e atividade atual. Manter **uma variável por grupo** evita dar peso duplicado à mesma informação.
3. **Antiguidade.** É o fator escondido por trás das variáveis acumuladas. Devemos preferir **taxas por unidade de tempo** aos totais.
4. **Perspetivas.** Os grupos encaixam nas perspetivas de segmentação pedidas no enunciado:
   - **Valor:** valor por mês de membro, plano, renovações.
   - **Comportamento:** intensidade de visitas, utilização do plano, recência.
   - **Serviços:** add-ons (`addon_diversity_score` e quantidades).
5. **Demografia.** Idade, género, país, indústria e conta de equipa quase não se relacionam com comportamento ou valor. Devem ser usadas para **descrever** os clusters finais, não para os criar.

---

## 9. Limitações

- **O Spearman só capta relações monótonas.** Uma relação em forma de U (por exemplo, valores altos nos dois extremos) teria ρ perto de 0.
- **Correlação não é causalidade.** As relações encontradas mostram associação, não causa. O passo 4 mostra um exemplo concreto de relação criada por uma terceira variável.
- **Pairwise deletion.** Cada correlação usa as linhas válidas desse par, por isso o número de linhas varia ligeiramente entre pares. A correlação parcial é calculada a partir dessa matriz, o que pode introduzir pequenas inconsistências.
- **Só se controlou a antiguidade.** Pode haver outros fatores de confusão, como o plano ou a coorte de entrada (a secção 1.3.2.5 mostra que o tamanho dos planos varia entre coortes).
- **A antiguidade também tem problemas de qualidade.** Cerca de 2,4% das linhas têm datas inconsistentes (secção 1.4.3), o que afeta `membership_duration_days`.
- **Muitos zeros.** As variáveis de add-ons são dominadas por zeros, o que pode subestimar a sua relação com as outras.

---

## 10. Perguntas prováveis na discussão

**Porquê Spearman e não Pearson?**
Os dados são muito assimétricos, têm muitos zeros e outliers extremos. O Spearman usa a ordem dos valores, por isso não é dominado por poucos valores extremos e capta relações monótonas que não são lineares.

**Correlação implica causalidade?**
Não. O passo 4 mostra isso mesmo: visitas totais e valor pareciam bastante relacionados (0,61), mas a relação era sobretudo efeito da antiguidade. Controlando a antiguidade, desce para 0,19.

**Porquê o limite de 0,8 para as redundâncias?**
É uma regra prática habitual para identificar variáveis que dão quase a mesma informação. A escolha final das variáveis faz-se na fase de clustering, de acordo com as perspetivas escolhidas.

**Porque não apagaram ou imputaram os valores inválidos?**
Para a correlação basta marcá-los como "em falta". Cada correlação usa todas as linhas válidas desse par, por isso não se perdem linhas inteiras. A decisão de como imputar fica para a fase de preparação dos dados, que é onde deve ser tomada.

**Porque usaram Cramér's V com correção de viés?**
A versão simples sobrestima a associação quando há muitas categorias (`country` e `industry` têm 8 e 10). A correção dá um valor mais honesto.

**Porque não usaram os p-values?**
Com mais de 15 mil linhas, quase qualquer associação, mesmo mínima, dá estatisticamente significativa. O que interessa para a segmentação é o tamanho da relação, não se ela existe.

**O que fariam a seguir?**
Escolher uma variável por grupo redundante, substituir os totais por taxas por tempo (por exemplo, valor por mês) e usar as variáveis demográficas apenas para perfilar os clusters.

**Há algo estranho nos dados?**
Sim. Os membros ativos visitaram menos no último período do que os que saíram, o que vai contra o esperado. Vale a pena investigar quando as datas forem limpas.
