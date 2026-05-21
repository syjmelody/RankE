# RankE — Project Page

Source for the project page of:

**RankE: End-to-End Post-Training for Discrete Text-to-Image Generation with Decoder Co-Evolution**
Siyong Jian, Siyuan Li, Luyuan Zhang, Zedong Wang, Xin Jin, Ying Li, Cheng Tan, Huan Wang.

## Sections

1. Abstract
2. The Problem: Latent Covariate Shift
3. The RankE Framework
4. Results (CLIP / HPSv2 quantitative + training dynamics)
5. Qualitative Comparisons
6. Citation (BibTeX)

## Structure

```
.
├── index.html         # main project page
├── style.css          # all styles
├── assets/            # converted figures (PNG)
│   ├── teaser1_intro.png
│   ├── fig_chap3_method.png
│   ├── combined_bar_pareto.png
│   ├── exp_result_visualize.png
│   └── llamagen_comparison.png
└── README.md
```

## Local preview

Open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```