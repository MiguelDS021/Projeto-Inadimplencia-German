German Credit Data — Classificação de Risco de Crédito

Projeto de classificação binária para prever o risco de crédito (bom ou mau pagador) utilizando o dataset Statlog (German Credit Data), da UCI Machine Learning Repository.

📌 Sobre o Dataset
Fonte: Hans Hofmann, Universität Hamburg (1994) — UCI ML Repository
Instâncias: 1.000
Atributos: 20 (7 numéricos + 13 categóricos)
Variável alvo (class): 1 = Good (bom pagador) · 2 = Bad (mau pagador)
Licença: CC BY 4.0
Matriz de custo

O problema exige avaliação assimétrica: classificar um mau pagador como bom custa 5x mais do que o erro inverso.

Previsto Good	Previsto Bad
Real Good	0	1
Real Bad	5	0

🗂️ Estrutura do pipeline
Encoding — tradução dos códigos originais (A11, A30, A61...) via dicionário aninhado
Mapeamento ordinal — conversão de checking_account, credit_history, savings, employment para escala numérica (0,1,2,3...), preservando ordem
Análise de outliers — regra do IQR nas variáveis duration, credit_amount, age
Correlação — Pearson (numéricas) e Cramér's V (categóricas) com o target
Seleção de features — remoção por baixa correlação, multicolinearidade e viés ético
One-Hot Encoding — variáveis nominais (purpose, property, other_debtors, other_installment_plans)
Split treino/teste — estratificado (stratify=y), 80/20
Padronização — StandardScaler, fit apenas no treino
Modelagem — Regressão Logística
Otimização — ajuste de threshold e class_weight

⚖️ Decisões éticas
personal_status_sex removida da classificação: o atributo mistura estado civil e sexo de forma desigual entre gêneros (categoria A92 agrupa "divorciada/separada/casada" apenas para mulheres, enquanto homens têm categorias distintas), o que introduziria viés discriminatório de gênero.
foreign_worker removida: associação estatística quase nula com o target (Cramér's V = 0,076) e variável sensível do ponto de vista de discriminação por nacionalidade.

🔎 Principais achados — Outliers
Variável	Nº outliers	% Bad no grupo	% Bad no dataset geral
credit_amount	72	54,2%	~30%
duration	70	57,1%	~30%
age	17	35,3%	~30%

Outliers de credit_amount e duration foram mantidos: concentram-se em finalidades legítimas (veículos, negócios) e carregam sinal preditivo relevante — a proporção de maus pagadores quase dobra nesses grupos.

🔗 Principais achados — Correlação

Numéricas (Pearson) com class:

Variável	Correlação
duration	0,215
credit_amount	0,155
age	-0,091

Categóricas (Cramér's V) com class:

Variável	Associação
checking_account	0,352
credit_history	0,248
savings	0,190

Multicolinearidade identificada: duration × credit_amount = 0,625 → credit_amount removida do modelo final, mantendo duration (maior correlação individual com o target).

🤖 Modelo e Resultados

Modelo base: Regressão Logística (sklearn.linear_model.LogisticRegression).

Três abordagens de otimização foram testadas para lidar com o desbalanceamento (70/30) e a matriz de custo assimétrica:

Abordagem	Recall (Bad)	Accuracy	Custo total
Threshold padrão (0.5)	0,62	0,60	172
Threshold otimizado (0.09)	0,78	0,52	148
class_weight='balanced'	0,70	0,54	165
class_weight={1:1, 2:5}	0,83	0,49	142

✅ Modelo final escolhido
python
LogisticRegression(max_iter=1000, random_state=0, class_weight={1: 1, 2: 5})

O peso {1: 1, 2: 5} foi definido a partir da própria matriz de custo do problema (não um balanceamento genérico), permitindo ao modelo aprender, já durante o treino, a penalizar mais fortemente o erro de classificar um mau pagador como bom. Essa abordagem superou tanto o ajuste de threshold isolado quanto o balanceamento automático ('balanced'), tanto em custo total quanto em recall da classe de risco.

⚠️ Limitação do resultado

O modelo final, embora minimize o custo segundo a matriz fornecida, recusa crédito a uma proporção alta de bons pagadores (92 de 140 no conjunto de teste). Esse trade-off é aceitável dentro da matriz de custo simplificada do dataset, mas um cenário real deveria considerar custos adicionais não capturados aqui (perda de receita de juros, relacionamento bancário, reputação) ao definir o ponto de operação final.

🛠️ Tecnologias
Python
pandas, numpy
scikit-learn
matplotlib, seaborn
scipy (teste qui-quadrado)

└── README.md
📚 Referências
Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
