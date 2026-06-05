# Diagramas UML — Cap. 3 do Relatório

Esta pasta contém os 4 diagramas UML pedidos pela "Proposta de Estrutura para o Relatório Final – LPEI" (capítulo 3 — *Engenharia de Requisitos e Modelação*).

| # | Diagrama | Ficheiro | Capítulo do relatório |
|---|----------|----------|------------------------|
| 1 | **Casos de Uso** | [01_use_cases.md](01_use_cases.md) | 3.2 — Modelação Funcional |
| 2 | **Atividade** | [02_activity.md](02_activity.md) | 3.3 — Modelação Comportamental |
| 3 | **Estados** | [03_state.md](03_state.md) | 3.3 — Modelação Comportamental |
| 4 | **Classes** | [04_class.md](04_class.md) | 3.4 — Modelação de Dados |

## Formato escolhido: Mermaid

Optei por **Mermaid** em vez de PlantUML/Draw.io porque:

1. **Render nativo**: GitHub, VS Code, Notion, GitLab, e a maioria dos visualizadores Markdown renderizam os diagramas inline sem ferramentas externas.
2. **Versionamento limpo**: o diagrama é texto. Diffs em pull requests mostram alterações estruturais legíveis.
3. **Sincronização com código**: quando uma classe muda, basta editar texto — não há ficheiros binários a regenerar.
4. **Exportação para o relatório**: cada diagrama pode ser exportado para PNG/SVG via:
   - **VS Code**: extensão *Markdown Preview Mermaid Support* → screenshot
   - **CLI**: `npx -p @mermaid-js/mermaid-cli mmdc -i 01_use_cases.md -o 01.png`
   - **Online**: https://mermaid.live (paste do código)

## Como exportar os diagramas para o relatório LaTeX/Word

### Via CLI (recomendado, reprodutível)

```bash
# Instalar uma vez:
npm install -g @mermaid-js/mermaid-cli

# Extrair apenas o bloco mermaid de um ficheiro e renderizar:
# (manual para já — script futuro em scripts/export_uml.sh)
mmdc -i docs/uml/01_use_cases.md -o docs/figures/uml_01_use_cases.png -w 1600
mmdc -i docs/uml/02_activity.md   -o docs/figures/uml_02_activity.png   -w 1600
mmdc -i docs/uml/03_state.md      -o docs/figures/uml_03_state.png      -w 1600
mmdc -i docs/uml/04_class.md      -o docs/figures/uml_04_class.png      -w 2400
```

> ⚠️ `mmdc` exporta o **primeiro bloco** mermaid encontrado no ficheiro. Cada um dos 4 ficheiros UML tem exatamente um bloco, por isso funciona out-of-the-box.

### Via online (rápido para um screenshot único)

1. Abre https://mermaid.live
2. Cola o conteúdo entre ` ```mermaid ` e ` ``` `
3. Click *Actions → PNG / SVG*
4. Guarda em `docs/figures/uml_*.png`

## Integração no relatório

Estas imagens devem ser referenciadas nos capítulos 3.2-3.4:

```latex
\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{figures/uml_01_use_cases.png}
  \caption{Diagrama de Casos de Uso do Deepfake Forensics Engine.}
  \label{fig:uml-use-cases}
\end{figure}
```

## Manutenção

Quando o código mudar:
- **Novo plugin** → adicionar à classe `PluginNames` (constante) e à tabela `SCENE_PLUGIN_WEIGHTS` → atualizar [04_class.md](04_class.md).
- **Novo endpoint** → atualizar `FastAPIApp` em [04_class.md](04_class.md) + caso de uso em [01_use_cases.md](01_use_cases.md).
- **Nova fase no pipeline** → atualizar [02_activity.md](02_activity.md) e [03_state.md](03_state.md).
