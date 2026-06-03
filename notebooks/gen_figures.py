#!/usr/bin/env python3
"""Run the EDA notebook and save all figures as PNG files."""
import json, sys, os
sys.path.insert(0, '..')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Style
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# Paths
BASE = Path('..')
BATCHES_PATH = BASE / 'xfacta_csv' / 'batches.csv'
BATCHES_CLEAN_PATH = BASE / 'xfacta_csv' / 'processed' / 'batches_clean.csv'
DEV_TEST_PATH = BASE / 'xfacta_csv' / 'dev_test.csv'
DEV_TEST_CLEAN_PATH = BASE / 'xfacta_csv' / 'processed' / 'dev_test_clean.csv'
DEV_JSON_PATH = BASE / 'xfacta_data' / 'dev.json'
TEST_JSON_PATH = BASE / 'xfacta_data' / 'test.json'
FIG_DIR = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

# Load data
batches_raw = pd.read_csv(BATCHES_PATH, encoding='utf-8-sig')
batches = pd.read_csv(BATCHES_CLEAN_PATH, encoding='utf-8-sig')
dev_test_raw = pd.read_csv(DEV_TEST_PATH, encoding='utf-8-sig')
dev_test = pd.read_csv(DEV_TEST_CLEAN_PATH, encoding='utf-8-sig')
with open(DEV_JSON_PATH) as f:
    dev_json = json.load(f)
with open(TEST_JSON_PATH) as f:
    test_json = json.load(f)

# Normalize label to string for consistent comparisons
batches['label'] = batches['label'].astype(str)

# Common columns
dev_test['text_len'] = dev_test['text'].fillna('').str.len()
batches['text_len'] = batches['text'].fillna('').str.len()
fake_only = batches[batches['sample_type'] == 'fake_sample'].copy()

# ========================================================
# Figure 2.1 — Label (True/False) Distribution Across Datasets
# ========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#2ecc71', '#e74c3c']

label_counts_b = batches['label'].value_counts()
axes[0].bar(label_counts_b.index, label_counts_b.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Label Distribution (batches_clean)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
for i, v in enumerate(label_counts_b.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

label_counts_dt = dev_test['label'].value_counts()
axes[1].bar(label_counts_dt.index, label_counts_dt.values, color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_title('Label Distribution (dev_test_clean)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Count')
for i, v in enumerate(label_counts_dt.values):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

dev_labels = pd.Series([str(x['label']) for x in dev_json]).value_counts()
axes[2].bar(dev_labels.index, dev_labels.values, color=colors[:len(dev_labels)], edgecolor='white', linewidth=1.5)
axes[2].set_title('Label Distribution (dev.json)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Label')
axes[2].set_ylabel('Count')
for i, v in enumerate(dev_labels.values):
    axes[2].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.suptitle('Figure 2.1 — Label (True/False) Distribution Across Datasets', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_2_1_label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_1_label_distribution.png')

# ========================================================
# Figure 2.2 — Distribution of Real vs Fake Samples
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sample_counts = batches['sample_type'].value_counts()
axes[0].bar(sample_counts.index, sample_counts.values, color=['#3498db', '#e74c3c'],
            edgecolor='white', linewidth=1.5)
axes[0].set_title('Sample Type (batches_clean)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sample Type')
axes[0].set_ylabel('Count')
for i, v in enumerate(sample_counts.values):
    axes[0].text(i, v + 10, f'{v} ({v/sample_counts.sum()*100:.1f}%)', ha='center', fontweight='bold')

axes[1].pie(sample_counts.values, labels=sample_counts.index, autopct='%1.1f%%',
           colors=['#3498db', '#e74c3c'], startangle=90, explode=(0.03, 0.03),
           textprops={'fontweight': 'bold'})
axes[1].set_title('Sample Type Proportions', fontsize=13, fontweight='bold')

plt.suptitle('Figure 2.2 — Distribution of Real vs Fake Samples', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_2_2_sample_type_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_2_sample_type.png')

# ========================================================
# Figure 2.3 — Misinformation Error Categories
# ========================================================
fig, ax = plt.subplots(figsize=(12, 6))
error_counts = fake_only['error_category'].value_counts()
bars = ax.barh(range(len(error_counts)), error_counts.values,
               color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(error_counts))),
               edgecolor='white', linewidth=1.2)
ax.set_yticks(range(len(error_counts)))
ax.set_yticklabels(error_counts.index, fontsize=11)
ax.set_xlabel('Count', fontsize=12)
ax.set_title('Error Category Distribution (Fake Samples Only)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(error_counts.values):
    ax.text(v + 3, i, str(v), va='center', fontweight='bold', fontsize=10)
total_fake = len(fake_only)
for i, v in enumerate(error_counts.values):
    pct = v / total_fake * 100
    ax.text(v + 60, i, f'({pct:.1f}%)', va='center', fontsize=9, color='gray')
plt.tight_layout()
plt.suptitle('Figure 2.3 — Misinformation Error Categories', fontsize=15, fontweight='bold', y=1.02)
plt.savefig(FIG_DIR / 'figure_2_3_error_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_3_error_categories.png')

# ========================================================
# Figure 2.4 — Topic Distribution: All vs Fake Samples
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
topic_counts = batches['topic'].value_counts()
colors_t = plt.cm.tab20(np.linspace(0, 1, len(topic_counts)))
axes[0].barh(range(len(topic_counts)), topic_counts.values, color=colors_t, edgecolor='white', linewidth=1.2)
axes[0].set_yticks(range(len(topic_counts)))
axes[0].set_yticklabels(topic_counts.index, fontsize=10)
axes[0].set_xlabel('Count', fontsize=12)
axes[0].set_title('Topic Distribution (All Samples)', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
for i, v in enumerate(topic_counts.values):
    axes[0].text(v + 10, i, str(v), va='center', fontweight='bold', fontsize=9)

fake_topic = fake_only['topic'].value_counts()
axes[1].barh(range(len(fake_topic)), fake_topic.values, color='#e74c3c', alpha=0.7, edgecolor='white', linewidth=1.2)
axes[1].set_yticks(range(len(fake_topic)))
axes[1].set_yticklabels(fake_topic.index, fontsize=10)
axes[1].set_xlabel('Count', fontsize=12)
axes[1].set_title('Topic Distribution (Fake Samples Only)', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
for i, v in enumerate(fake_topic.values):
    axes[1].text(v + 3, i, str(v), va='center', fontweight='bold', fontsize=9)

plt.suptitle('Figure 2.4 — Topic Distribution: All vs Fake Samples', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_2_4_topic_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_4_topic_distribution.png')

# ========================================================
# Figure 2.5 — Text Length Distribution
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for label, color in [('True', '#2ecc71'), ('False', '#e74c3c')]:
    data = batches[batches['label'] == label]['text_len'].dropna()
    axes[0].hist(data, bins=60, alpha=0.6, color=color, label=f'{label} (n={len(data)})', density=True)
axes[0].set_xlabel('Text Length (characters)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Text Length by Label (batches_clean)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)

for st, color in [('real_sample', '#3498db'), ('fake_sample', '#e74c3c')]:
    data = batches[batches['sample_type'] == st]['text_len'].dropna()
    axes[1].hist(data, bins=60, alpha=0.6, color=color, label=f'{st} (n={len(data)})', density=True)
axes[1].set_xlabel('Text Length (characters)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Text Length by Sample Type (batches_clean)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)

plt.suptitle('Figure 2.5 — Text Length Distribution', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_2_5_text_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_5_text_length.png')

# ========================================================
# Figure 3.1 — Label Distribution Within Each Error Category
# ========================================================
fig, ax = plt.subplots(figsize=(12, 6))
ct_error = pd.crosstab(batches['error_category'], batches['label'], normalize='index') * 100
ct_error.columns = ct_error.columns.astype(str)
ct_error = ct_error.sort_values('False', ascending=True)
ct_error.plot(kind='barh', stacked=True, color=['#2ecc71', '#e74c3c'], ax=ax, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Percentage (%)', fontsize=12)
ax.set_ylabel('Error Category', fontsize=12)
ax.set_title('Figure 3.1 — Label Distribution Within Each Error Category', fontsize=14, fontweight='bold')
ax.legend(title='Label', fontsize=11, loc='lower right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_3_1_label_by_error_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_1_error_category_label.png')

# ========================================================
# Figure 3.2 — Label Percentage by Topic
# ========================================================
fig, ax = plt.subplots(figsize=(14, 8))
ct_topic = pd.crosstab(batches['topic'], batches['label'], normalize='index') * 100
false_col = 'False' if 'False' in ct_topic.columns else ct_topic.columns[0]
ct_topic = ct_topic.sort_values(false_col, ascending=False)
sns.heatmap(ct_topic, annot=True, fmt='.1f', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, cbar_kws={'label': '% within Topic'})
ax.set_title('Figure 3.2 — Label Percentage by Topic', fontsize=14, fontweight='bold')
ax.set_xlabel('Label', fontsize=12)
ax.set_ylabel('Topic', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_3_2_label_by_topic.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_2_topic_label_heatmap.png')

# ========================================================
# Figure 3.3 — Image Presence & Count vs Label
# ========================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
batches['has_image'] = batches['images'].notna() & (batches['images'] != '')
img_presence = pd.crosstab(batches['has_image'], batches['label'], normalize='index') * 100
img_presence.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=axes[0], edgecolor='white')
axes[0].set_title('Label: With vs Without Images', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Has Image?')
axes[0].set_ylabel('Percentage')
axes[0].legend(title='Label')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['No Image', 'Has Image'])

batches['num_images'] = batches['images'].fillna('').apply(lambda x: len(x.split('; ')) if x else 0)
img_counts = batches[batches['num_images'] > 0]['num_images'].copy()
img_counts_clipped = img_counts.clip(upper=5)
ct_multi = pd.crosstab(img_counts_clipped, batches.loc[img_counts.index, 'label'])
ct_multi_pct = ct_multi.div(ct_multi.sum(axis=1), axis=0) * 100
ct_multi_pct.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=axes[1], edgecolor='white')
axes[1].set_title('Label by Number of Images', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Images')
axes[1].set_ylabel('Percentage')
axes[1].legend(title='Label')

plt.suptitle('Figure 3.3 — Image Presence & Count vs Label', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_3_3_image_presence_and_count.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_3_image_presence.png')

# ========================================================
# Figure 3.4 — Author Type Analysis
# ========================================================
known_orgs = ['CNN', 'BBC News', 'BBC News (World)', 'Fox News', 'BBC Breaking News',
              'The Guardian', 'Washington Post', 'Reuters']
batches['author_type'] = batches['author'].apply(
    lambda a: 'News Organization' if a in known_orgs else
             ('Individual' if str(a) != 'Unknown' else 'Unknown'))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
author_type_counts = batches['author_type'].value_counts()
axes[0].bar(author_type_counts.index, author_type_counts.values,
           color=['#3498db', '#e74c3c', '#95a5a6'], edgecolor='white', linewidth=1.5)
axes[0].set_title('Author Type Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Author Type')
axes[0].set_ylabel('Count')
for i, v in enumerate(author_type_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

ct_author = pd.crosstab(batches['author_type'], batches['label'], normalize='index') * 100
ct_author.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=axes[1], edgecolor='white')
axes[1].set_title('Label Distribution by Author Type', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Author Type')
axes[1].set_ylabel('Percentage')
axes[1].legend(title='Label')

plt.suptitle('Figure 3.4 — Author Type Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_3_4_author_type_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_4_author_type.png')

# ========================================================
# Figure 4.1 — % Fake by Topic and Error Category
# ========================================================
fake_prop = batches[batches['error_category'].notna() & (batches['error_category'] != '')].copy()
fake_prop['is_fake'] = (fake_prop['label'] == 'False').astype(int)
pivot = fake_prop.pivot_table(values='is_fake', index='topic', columns='error_category', aggfunc='mean') * 100
pivot = pivot.dropna(thresh=3)

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, cbar_kws={'label': '% Fake'},
            vmin=0, vmax=100)
ax.set_title('Figure 4.1 — % Fake by Topic and Error Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Error Category', fontsize=12)
ax.set_ylabel('Topic', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_4_1_fake_by_topic_and_error_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_1_topic_error_heatmap.png')

# ========================================================
# Figure 4.2 — Misinformation Timeline by Label
# ========================================================
batches['date_posted'] = pd.to_datetime(batches['date_posted'], errors='coerce')
timeline = batches.dropna(subset=['date_posted']).copy()
timeline['month'] = timeline['date_posted'].dt.to_period('M')

fig, ax = plt.subplots(figsize=(16, 6))
monthly_counts = timeline.groupby(['month', 'label']).size().unstack(fill_value=0)
monthly_counts.index = monthly_counts.index.astype(str)

ax.plot(monthly_counts.index, monthly_counts.get('True', pd.Series(0, index=monthly_counts.index)),
        'o-', color='#2ecc71', linewidth=2, markersize=6, label='True (Real)')
ax.plot(monthly_counts.index, monthly_counts.get('False', pd.Series(0, index=monthly_counts.index)),
        's-', color='#e74c3c', linewidth=2, markersize=6, label='False (Fake)')
ax.fill_between(monthly_counts.index,
                monthly_counts.get('False', pd.Series(0, index=monthly_counts.index)),
                alpha=0.1, color='#e74c3c')

ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Posts', fontsize=12)
ax.set_title('Figure 4.2 — Misinformation Timeline by Label', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.tick_params(axis='x', rotation=45)
ax.axvline(x='2024-11', color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text('2024-11', ax.get_ylim()[1]*0.95, 'US Election', color='orange', fontsize=10, ha='center')

plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_4_2_timeline_by_label.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_2_timeline_label.png')

# ========================================================
# Figure 4.3 — Fake Post Timeline by Error Category
# ========================================================
fig, ax = plt.subplots(figsize=(16, 6))
fake_timeline = timeline[timeline['label'] == 'False'].copy()
error_monthly = fake_timeline.groupby(['month', 'error_category']).size().unstack(fill_value=0).astype(float)
error_monthly.index = error_monthly.index.astype(str)
error_monthly.plot(ax=ax, marker='o', markersize=4, linewidth=1.5, alpha=0.8)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Fake Posts', fontsize=12)
ax.set_title('Figure 4.3 — Fake Post Timeline by Error Category', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.tick_params(axis='x', rotation=45)
ax.axvline(x='2024-11', color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text('2024-11', ax.get_ylim()[1]*0.95, 'US Election', color='orange', fontsize=10, ha='center')

plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_4_3_timeline_by_error_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_3_timeline_error_category.png')

# ========================================================
# Figure 4.4 — Text Length by Topic and Label
# ========================================================
top_topics = batches['topic'].value_counts().head(10).index
plot_df = batches[batches['topic'].isin(top_topics) & batches['label'].notna()].copy()

fig, ax = plt.subplots(figsize=(16, 7))
sns.boxplot(data=plot_df, x='topic', y='text_len', hue='label',
            palette={'True': '#2ecc71', 'False': '#e74c3c'}, ax=ax,
            linewidth=1.2, fliersize=3)
ax.set_xlabel('Topic', fontsize=12)
ax.set_ylabel('Text Length (characters)', fontsize=12)
ax.set_title('Figure 4.4 — Text Length by Topic and Label', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.savefig(FIG_DIR / 'figure_4_4_text_length_by_topic_and_label.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_4_text_length_boxplot.png')

print(f'\nAll figures saved to {FIG_DIR.resolve()}')
