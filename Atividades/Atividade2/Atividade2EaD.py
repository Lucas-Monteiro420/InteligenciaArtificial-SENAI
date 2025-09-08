# Controle de Qualidade - Análise Estatística de Lote de Peças
# Fundamentos da Estatística Aplicados à Qualidade

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Tentativa de importar seaborn (opcional)
try:
    import seaborn as sns
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    print("Seaborn não disponível. Usando matplotlib padrão.")
    SEABORN_AVAILABLE = False

# Configuração para gráficos
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
        print("Usando estilo padrão do matplotlib")

print("="*60)
print("CONTROLE DE QUALIDADE - ANÁLISE ESTATÍSTICA")
print("Lote: 5000 peças produzidas")
print("Variables analisadas: Peso e Dimensões")
print("="*60)

## 1. SIMULAÇÃO DOS DADOS POPULACIONAIS

# Definindo parâmetros populacionais das peças
np.random.seed(42)

# Peso das peças (em gramas) - distribuição normal
peso_populacional = np.random.normal(loc=150, scale=5, size=5000)

# Dimensão das peças (em mm) - distribuição normal
dimensao_populacional = np.random.normal(loc=25.0, scale=0.8, size=5000)

# Criando DataFrame populacional
populacao = pd.DataFrame({
    'peso': peso_populacional,
    'dimensao': dimensao_populacional
})

print(f"População total: {len(populacao)} peças")
print(f"Peso médio populacional: {populacao['peso'].mean():.2f}g")
print(f"Dimensão média populacional: {populacao['dimensao'].mean():.2f}mm")

## 2. CONCEITOS DE AMOSTRAGEM

print("\n" + "="*50)
print("2. PROCESSO DE AMOSTRAGEM")
print("="*50)

# Definindo critérios de especificação do produto
especificacoes = {
    'peso_min': 140,      # gramas
    'peso_max': 160,      # gramas
    'dimensao_min': 23.5, # mm
    'dimensao_max': 26.5  # mm
}

print("Especificações do produto:")
for spec, valor in especificacoes.items():
    print(f"  {spec}: {valor}")

# Cálculo do tamanho da amostra
# Usando nível de confiança de 95% e margem de erro de 2%
N = 5000  # tamanho da população
z = 1.96  # valor crítico para 95% de confiança
p = 0.5   # proporção estimada (mais conservadora)
E = 0.02  # margem de erro

n_calculado = (z**2 * p * (1-p) * N) / (E**2 * (N-1) + z**2 * p * (1-p))
n_amostra = int(np.ceil(n_calculado))

print(f"\nCálculo do tamanho da amostra:")
print(f"  População (N): {N}")
print(f"  Nível de confiança: 95% (z = {z})")
print(f"  Margem de erro: {E*100}%")
print(f"  Tamanho da amostra calculado: {n_amostra} peças")

# Amostragem aleatória simples
indices_amostra = np.random.choice(len(populacao), size=n_amostra, replace=False)
amostra = populacao.iloc[indices_amostra].copy()

print(f"\nAmostra coletada: {len(amostra)} peças")
print("Método: Amostragem Aleatória Simples")

## 3. MEDIÇÕES DE TENDÊNCIA CENTRAL E DISPERSÃO

print("\n" + "="*50)
print("3. ANÁLISE DESCRITIVA DA AMOSTRA")
print("="*50)

def calcular_estatisticas(dados, variavel):
    """Calcula estatísticas descritivas completas"""
    
    stats_dict = {
        'n': len(dados),
        'média': dados.mean(),
        'mediana': dados.median(),
        'moda': dados.mode().iloc[0] if not dados.mode().empty else 'N/A',
        'desvio_padrão': dados.std(ddof=1),  # amostra
        'variância': dados.var(ddof=1),       # amostra
        'coef_variação': (dados.std(ddof=1) / dados.mean()) * 100,
        'amplitude': dados.max() - dados.min(),
        'quartil_1': dados.quantile(0.25),
        'quartil_3': dados.quantile(0.75),
        'amplitude_interquartil': dados.quantile(0.75) - dados.quantile(0.25),
        'assimetria': stats.skew(dados),
        'curtose': stats.kurtosis(dados)
    }
    
    return stats_dict

# Calculando estatísticas para peso
stats_peso = calcular_estatisticas(amostra['peso'], 'peso')
stats_dimensao = calcular_estatisticas(amostra['dimensao'], 'dimensao')

print("ESTATÍSTICAS DESCRITIVAS - PESO (gramas)")
print("-" * 40)
for stat, valor in stats_peso.items():
    if isinstance(valor, (int, float)):
        print(f"{stat.capitalize()}: {valor:.3f}")
    else:
        print(f"{stat.capitalize()}: {valor}")

print("\nESTATÍSTICAS DESCRITIVAS - DIMENSÃO (mm)")
print("-" * 40)
for stat, valor in stats_dimensao.items():
    if isinstance(valor, (int, float)):
        print(f"{stat.capitalize()}: {valor:.3f}")
    else:
        print(f"{stat.capitalize()}: {valor}")

# Criando gráficos de análise descritiva
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análise Descritiva da Amostra', fontsize=16, fontweight='bold')

# Histogramas
axes[0,0].hist(amostra['peso'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(stats_peso['média'], color='red', linestyle='--', label=f'Média: {stats_peso["média"]:.2f}')
axes[0,0].axvline(especificacoes['peso_min'], color='orange', linestyle=':', label='Limite Inferior')
axes[0,0].axvline(especificacoes['peso_max'], color='orange', linestyle=':', label='Limite Superior')
axes[0,0].set_title('Distribuição do Peso')
axes[0,0].set_xlabel('Peso (g)')
axes[0,0].set_ylabel('Frequência')
axes[0,0].legend()

axes[1,0].hist(amostra['dimensao'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1,0].axvline(stats_dimensao['média'], color='red', linestyle='--', label=f'Média: {stats_dimensao["média"]:.2f}')
axes[1,0].axvline(especificacoes['dimensao_min'], color='orange', linestyle=':', label='Limite Inferior')
axes[1,0].axvline(especificacoes['dimensao_max'], color='orange', linestyle=':', label='Limite Superior')
axes[1,0].set_title('Distribuição da Dimensão')
axes[1,0].set_xlabel('Dimensão (mm)')
axes[1,0].set_ylabel('Frequência')
axes[1,0].legend()

# Boxplots
axes[0,1].boxplot(amostra['peso'])
axes[0,1].set_title('Boxplot - Peso')
axes[0,1].set_ylabel('Peso (g)')

axes[1,1].boxplot(amostra['dimensao'])
axes[1,1].set_title('Boxplot - Dimensão')
axes[1,1].set_ylabel('Dimensão (mm)')

# Q-Q plots para normalidade
stats.probplot(amostra['peso'], dist="norm", plot=axes[0,2])
axes[0,2].set_title('Q-Q Plot - Peso')

stats.probplot(amostra['dimensao'], dist="norm", plot=axes[1,2])
axes[1,2].set_title('Q-Q Plot - Dimensão')

plt.tight_layout()
plt.show()

## 4. TESTE DE HIPÓTESES

print("\n" + "="*50)
print("4. TESTE DE HIPÓTESES")
print("="*50)

# Definindo as hipóteses para cada variável

print("TESTE 1: PESO DAS PEÇAS")
print("-" * 30)
print("H0: μ_peso = 150g (o peso médio está conforme o padrão)")
print("H1: μ_peso ≠ 150g (o peso médio não está conforme o padrão)")
print("Nível de significância: α = 0.05")
print("Teste: bilateral (bicaudal)")

# Teste t para uma amostra - PESO
peso_target = 150
alpha = 0.05

t_stat_peso, p_value_peso = stats.ttest_1samp(amostra['peso'], peso_target)

print(f"\nResultados do teste - PESO:")
print(f"  Estatística t: {t_stat_peso:.4f}")
print(f"  Valor-p: {p_value_peso:.6f}")
print(f"  Graus de liberdade: {len(amostra)-1}")

# Região crítica
t_critico = stats.t.ppf(1-alpha/2, len(amostra)-1)
print(f"  Valor crítico (±): {t_critico:.4f}")

# Decisão
if p_value_peso < alpha:
    decisao_peso = "Rejeitamos H0"
    conclusao_peso = "O peso médio das peças NÃO está conforme o padrão"
else:
    decisao_peso = "Não rejeitamos H0"
    conclusao_peso = "O peso médio das peças está conforme o padrão"

print(f"  Decisão: {decisao_peso}")
print(f"  Conclusão: {conclusao_peso}")

print(f"\nTESTE 2: DIMENSÃO DAS PEÇAS")
print("-" * 30)
print("H0: μ_dimensão = 25.0mm (a dimensão média está conforme o padrão)")
print("H1: μ_dimensão ≠ 25.0mm (a dimensão média não está conforme o padrão)")
print("Nível de significância: α = 0.05")
print("Teste: bilateral (bicaudal)")

# Teste t para uma amostra - DIMENSÃO
dimensao_target = 25.0

t_stat_dimensao, p_value_dimensao = stats.ttest_1samp(amostra['dimensao'], dimensao_target)

print(f"\nResultados do teste - DIMENSÃO:")
print(f"  Estatística t: {t_stat_dimensao:.4f}")
print(f"  Valor-p: {p_value_dimensao:.6f}")
print(f"  Graus de liberdade: {len(amostra)-1}")
print(f"  Valor crítico (±): {t_critico:.4f}")

# Decisão
if p_value_dimensao < alpha:
    decisao_dimensao = "Rejeitamos H0"
    conclusao_dimensao = "A dimensão média das peças NÃO está conforme o padrão"
else:
    decisao_dimensao = "Não rejeitamos H0"
    conclusao_dimensao = "A dimensão média das peças está conforme o padrão"

print(f"  Decisão: {decisao_dimensao}")
print(f"  Conclusão: {conclusao_dimensao}")

# Teste adicional: Proporção de peças dentro das especificações
print(f"\nTESTE 3: CONFORMIDADE COM ESPECIFICAÇÕES")
print("-" * 45)

# Calculando conformidade
peso_conforme = ((amostra['peso'] >= especificacoes['peso_min']) & 
                 (amostra['peso'] <= especificacoes['peso_max']))
dimensao_conforme = ((amostra['dimensao'] >= especificacoes['dimensao_min']) & 
                     (amostra['dimensao'] <= especificacoes['dimensao_max']))

# Peças totalmente conformes (peso E dimensão dentro das specs)
pecas_conformes = peso_conforme & dimensao_conforme
proporcao_conformes = pecas_conformes.sum() / len(amostra)

print(f"Peças com peso conforme: {peso_conforme.sum()}/{len(amostra)} ({peso_conforme.mean()*100:.1f}%)")
print(f"Peças com dimensão conforme: {dimensao_conforme.sum()}/{len(amostra)} ({dimensao_conforme.mean()*100:.1f}%)")
print(f"Peças totalmente conformes: {pecas_conformes.sum()}/{len(amostra)} ({proporcao_conformes*100:.1f}%)")

# Teste de hipótese para proporção
print(f"\nH0: p ≥ 0.95 (pelo menos 95% das peças estão conformes)")
print(f"H1: p < 0.95 (menos de 95% das peças estão conformes)")
print("Teste: unilateral (unicaudal à esquerda)")

p0 = 0.95  # proporção esperada
n = len(amostra)
x = pecas_conformes.sum()  # sucessos observados
p_obs = x / n  # proporção observada

# Teste Z para proporção
z_stat = (p_obs - p0) / np.sqrt(p0 * (1 - p0) / n)
p_value_prop = stats.norm.cdf(z_stat)  # teste unicaudal à esquerda

print(f"\nResultados do teste - PROPORÇÃO:")
print(f"  Proporção observada: {p_obs:.4f}")
print(f"  Estatística Z: {z_stat:.4f}")
print(f"  Valor-p: {p_value_prop:.6f}")

z_critico = stats.norm.ppf(alpha)  # valor crítico para teste unicaudal
print(f"  Valor crítico: {z_critico:.4f}")

if p_value_prop < alpha:
    decisao_prop = "Rejeitamos H0"
    conclusao_prop = "MENOS de 95% das peças estão conformes"
else:
    decisao_prop = "Não rejeitamos H0"
    conclusao_prop = "PELO MENOS 95% das peças estão conformes"

print(f"  Decisão: {decisao_prop}")
print(f"  Conclusão: {conclusao_prop}")

## 5. INFERÊNCIA ESTATÍSTICA E INTERVALOS DE CONFIANÇA

print("\n" + "="*50)
print("5. INFERÊNCIA ESTATÍSTICA")
print("="*50)

# Intervalos de confiança para as médias
confianca = 0.95
alpha_ic = 1 - confianca

# IC para peso
erro_padrao_peso = stats_peso['desvio_padrão'] / np.sqrt(len(amostra))
margem_erro_peso = t_critico * erro_padrao_peso
ic_peso_inferior = stats_peso['média'] - margem_erro_peso
ic_peso_superior = stats_peso['média'] + margem_erro_peso

print(f"INTERVALO DE CONFIANÇA - PESO ({confianca*100}%)")
print(f"  Média amostral: {stats_peso['média']:.3f}g")
print(f"  Erro padrão: {erro_padrao_peso:.3f}")
print(f"  Margem de erro: ±{margem_erro_peso:.3f}")
print(f"  IC: [{ic_peso_inferior:.3f}, {ic_peso_superior:.3f}]")

# IC para dimensão
erro_padrao_dimensao = stats_dimensao['desvio_padrão'] / np.sqrt(len(amostra))
margem_erro_dimensao = t_critico * erro_padrao_dimensao
ic_dimensao_inferior = stats_dimensao['média'] - margem_erro_dimensao
ic_dimensao_superior = stats_dimensao['média'] + margem_erro_dimensao

print(f"\nINTERVALO DE CONFIANÇA - DIMENSÃO ({confianca*100}%)")
print(f"  Média amostral: {stats_dimensao['média']:.3f}mm")
print(f"  Erro padrão: {erro_padrao_dimensao:.3f}")
print(f"  Margem de erro: ±{margem_erro_dimensao:.3f}")
print(f"  IC: [{ic_dimensao_inferior:.3f}, {ic_dimensao_superior:.3f}]")

# IC para proporção
erro_padrao_prop = np.sqrt(p_obs * (1 - p_obs) / n)
z_ic = stats.norm.ppf(1 - alpha_ic/2)
margem_erro_prop = z_ic * erro_padrao_prop
ic_prop_inferior = max(0, p_obs - margem_erro_prop)
ic_prop_superior = min(1, p_obs + margem_erro_prop)

print(f"\nINTERVALO DE CONFIANÇA - PROPORÇÃO ({confianca*100}%)")
print(f"  Proporção amostral: {p_obs:.4f}")
print(f"  Erro padrão: {erro_padrao_prop:.4f}")
print(f"  Margem de erro: ±{margem_erro_prop:.4f}")
print(f"  IC: [{ic_prop_inferior:.4f}, {ic_prop_superior:.4f}]")

## 6. DECISÃO FINAL E RECOMENDAÇÕES

print("\n" + "="*60)
print("6. DECISÃO FINAL SOBRE O LOTE")
print("="*60)

# Critérios de aprovação
criterios_aprovacao = {
    'peso_media_ok': abs(t_stat_peso) <= t_critico,
    'dimensao_media_ok': abs(t_stat_dimensao) <= t_critico,
    'conformidade_ok': p_value_prop >= alpha
}

print("CRITÉRIOS DE APROVAÇÃO:")
print(f"  ✓ Peso médio conforme (|t| ≤ t_crítico): {criterios_aprovacao['peso_media_ok']}")
print(f"  ✓ Dimensão média conforme (|t| ≤ t_crítico): {criterios_aprovacao['dimensao_media_ok']}")
print(f"  ✓ Conformidade ≥ 95% (p-valor ≥ α): {criterios_aprovacao['conformidade_ok']}")

todos_criterios_ok = all(criterios_aprovacao.values())

print(f"\nDECISÃO FINAL:")
if todos_criterios_ok:
    decisao_final = "APROVADO"
    cor_decisao = "🟢"
else:
    decisao_final = "REPROVADO"
    cor_decisao = "🔴"

print(f"{cor_decisao} LOTE {decisao_final}")

print(f"\nJUSTIFICATIVA:")
if criterios_aprovacao['peso_media_ok']:
    print("  • Peso médio está dentro do padrão esperado")
else:
    print("  • Peso médio NÃO está dentro do padrão esperado")

if criterios_aprovacao['dimensao_media_ok']:
    print("  • Dimensão média está dentro do padrão esperado")
else:
    print("  • Dimensão média NÃO está dentro do padrão esperado")

if criterios_aprovacao['conformidade_ok']:
    print("  • Taxa de conformidade atende ao mínimo exigido (≥95%)")
else:
    print("  • Taxa de conformidade NÃO atende ao mínimo exigido (≥95%)")

print(f"\nRECOMENDAÇÕES:")
if decisao_final == "APROVADO":
    print("  1. O lote pode ser enviado ao cliente")
    print("  2. Manter monitoramento contínuo do processo")
    print("  3. Arquivar documentação para rastreabilidade")
else:
    print("  1. NÃO enviar o lote ao cliente")
    print("  2. Investigar causas das não-conformidades")
    print("  3. Implementar ações corretivas no processo")
    print("  4. Realizar nova amostragem após correções")

# Resumo estatístico final
print(f"\n" + "="*50)
print("RESUMO ESTATÍSTICO FINAL")
print("="*50)
print(f"Tamanho da amostra: {len(amostra)} peças")
print(f"Peso: μ = {stats_peso['média']:.2f}±{margem_erro_peso:.2f}g (IC 95%)")
print(f"Dimensão: μ = {stats_dimensao['média']:.2f}±{margem_erro_dimensao:.2f}mm (IC 95%)")
print(f"Conformidade: {proporcao_conformes*100:.1f}% das peças")
print(f"Status do lote: {decisao_final}")

# Gráfico final de conformidade
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de barras para conformidade
categorias = ['Peso\nConforme', 'Dimensão\nConforme', 'Totalmente\nConforme']
valores = [peso_conforme.mean()*100, dimensao_conforme.mean()*100, proporcao_conformes*100]
cores = ['lightblue', 'lightgreen', 'gold']

bars = ax1.bar(categorias, valores, color=cores, alpha=0.8, edgecolor='black')
ax1.axhline(y=95, color='red', linestyle='--', label='Mínimo Exigido (95%)')
ax1.set_ylabel('Percentual de Conformidade (%)')
ax1.set_title('Taxa de Conformidade por Critério')
ax1.set_ylim(0, 100)
ax1.legend()

# Adicionando valores nas barras
for bar, valor in zip(bars, valores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{valor:.1f}%', ha='center', va='bottom', fontweight='bold')

# Gráfico de dispersão peso vs dimensão
scatter = ax2.scatter(amostra['peso'], amostra['dimensao'], 
                     c=pecas_conformes, cmap='RdYlGn', alpha=0.6, s=50)
ax2.axvline(especificacoes['peso_min'], color='red', linestyle=':', alpha=0.7)
ax2.axvline(especificacoes['peso_max'], color='red', linestyle=':', alpha=0.7)
ax2.axhline(especificacoes['dimensao_min'], color='red', linestyle=':', alpha=0.7)
ax2.axhline(especificacoes['dimensao_max'], color='red', linestyle=':', alpha=0.7)
ax2.set_xlabel('Peso (g)')
ax2.set_ylabel('Dimensão (mm)')
ax2.set_title('Distribuição: Peso vs Dimensão')
plt.colorbar(scatter, ax=ax2, label='Conforme (1=Sim, 0=Não)')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ANÁLISE CONCLUÍDA")
print("="*60)