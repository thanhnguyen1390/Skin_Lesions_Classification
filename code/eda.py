import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

data_dir = "../data"
plot_dir = "../plots"
# output_dir = "../output"
metadata_file_name = "HAM10000_metadata.csv"

metadata = pd.read_csv(os.path.join(data_dir, metadata_file_name))

# Check for na values
print(metadata.isnull().sum())

# Replace null age with median
age_median = metadata.age.median()
metadata.age.fillna((metadata.age.mean()), inplace=True)

# Check for na values
print(metadata.isnull().sum())

# Lesion Type Dictionary
cell_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


metadata['cell_type'] = metadata.dx.map(cell_type_dict)
metadata['cell_type_idx'] = pd.Categorical(metadata['cell_type']).codes

metadata.to_csv(os.path.join(data_dir, 'HAM10000_metadata_cleaned.csv'))

###########################
#        PLOTS            #
###########################
plt.style.use(['classic'])
# Age
# metadata.age.plot(kind="hist", title="Age", fontsize=8)
# plt.savefig(os.path.join(plot_dir, "Age.png"), bbox_inches='tight')
# plt.clf()
#
# # DX Type
# metadata.dx_type.value_counts(sort=True, ascending=False).plot(kind="barh", title="Diagnosis Type")
# plt.savefig(os.path.join(plot_dir, "DX_type.png"), bbox_inches='tight')
# plt.clf()
#
# # Gender
# metadata.sex.value_counts(sort=True, ascending=False).plot(kind="barh", title="Gender", fontsize=8)
# plt.savefig(os.path.join(plot_dir, "Gender.png"), bbox_inches='tight')
# plt.clf()
#
# # Location
# metadata.localization.value_counts(sort=True, ascending=False).plot(kind="barh", title="Location", fontsize=8)
# plt.savefig(os.path.join(plot_dir, "Localization.png"), bbox_inches='tight')
# plt.clf()
#
# # Cell Type
# metadata.cell_type.value_counts(sort=True, ascending=False).plot(kind="barh", title="SKin Lesion Type", fontsize=8, rot= 0, figsize=(10,5))
# plt.savefig(os.path.join(plot_dir, "Skin_lesion_type.png"), bbox_inches='tight')
# plt.clf()
#
#
# # Sample Images
# samples = 5
# image_dir = '../data/images'
# fig, m_axs = plt.subplots(7, samples, figsize=(4*samples, 3*7))
# for n_axs, (type_name, type_rows) in zip(m_axs,
#                                          metadata.sort_values(['cell_type']).groupby('cell_type')):
#     n_axs[0].set_title(type_name)
#     for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(samples, random_state=13).iterrows()):
#         image = Image.open(os.path.join(image_dir, c_row.image_id + '.jpg'))
#         c_ax.imshow(image)
#         c_ax.axis('off')
# plt.savefig(os.path.join(plot_dir, "skin_lesion_samples.png"), bbox_inches='tight')
