import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# utilizei o gemini para entender o que fazer em cada passo da instruçao
# 1
df = pd.read_csv('medical_examination.csv')

bmi = df['weight'] / (df['height'] / 100) ** 2

# 2
df['overweight'] = (bmi > 25).astype(int)
print("Colum 'overweght' add:")
print(df[['weight', 'height', 'overweight']].head())

# 3 
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

print("\nClolums 'cholesterol' and 'gluc' are normalized:")
print("value count to cholesterol:")
print(df['cholesterol'].value_counts())
print("\nValue count to Gluc:")
print(df['gluc'].value_counts())


# 4
def draw_cat_plot():
    variables = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=variables)

    g = sns.catplot(data=df_cat, x='variable', kind='count', hue='value', col='cardio')
    
    # 8
    fig = g.fig


    # 9
    fig.savefig('catplot.png')
    return fig
#O que cada parâmetro do catplot faz:
#data=df_cat: Especifica nosso DataFrame reorganizado como a fonte dos dados.
#x='variable': Coloca o nome de cada variável (active, alco, etc.) no eixo X.
#kind='count': Diz ao Seaborn para criar um gráfico de barras contando quantas vezes cada value (0 ou 1) aparece para cada variable. Ele faz o trabalho de contagem para nós!
#hue='value': Cria barras separadas e coloridas para os valores 0 e 1 dentro de cada variável.
#col='cardio': Cria sub-gráficos distintos lado a lado, um para cada valor da coluna cardio. É isso que nos permite comparar visualmente o grupo saudável com o grupo doente.


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 10))


    # 15
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        linewidths=.5,
        square=True,
        center=0,
        vmax=0.32,
        cbar_kws={"shrink": 0.5}
    )



    # 16
    fig.savefig('heatmap.png')
    return fig

draw_cat_plot()
draw_heat_map()
