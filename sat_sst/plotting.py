import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def setup(font_dir='/content/drive/MyDrive/sst/Avenir'):
    for font in fm.findSystemFonts(font_dir):
        fm.fontManager.addfont(font)
    mpl.rcParams['font.family'] = 'Avenir'


def plot_model_data(data, i=None, save_name='test.png'):
    i = i or np.random.randint(0, data.data['target_sst'].shape[0])
    plot_data = {
        k: data.get(k, array_type='numpy')[i, 0]
        # k: data.data[k].detach().numpy()[i, 0]
        for k in ('input_sst', 'pred_sst', 'target_sst', 'target_mask')
    }
    plot_data['pred_sst'] = np.where(plot_data['target_mask'], plot_data['pred_sst'], plot_data['target_sst'])
    plot_data['input_sst'] = np.where(plot_data['target_mask'], np.nan, plot_data['input_sst'])
    plot_data['pred_mask'] = data.get('pred_mask', array_type='numpy')[i].argmax(axis=0)

    sst_kwargs = {
        'cmap': 'Spectral_r',
        'vmin': np.nanmin(plot_data['target_sst']),
        'vmax': np.nanmax(plot_data['target_sst']),
    }
    mask_kwargs = {
        'cmap': 'gray', 'vmin': 0, 'vmax': 1,
    }
    cols = ['input', 'pred', 'target']
    fig, axes = plt.subplots(2, 3, figsize=(3 * 2, 1.75 * 2))
    for i, c in enumerate(cols):
        sst = plot_data[f'{c}_sst']
        im = axes[0][i].imshow(sst, **sst_kwargs)
        fig.colorbar(im, ax=axes[0][i])
        axes[0][i].set_title(c)
        if c == 'input':
            axes[1][i].axis('off')
            continue
        mask = plot_data[f'{c}_mask']
        im = axes[1][i].imshow(mask, **mask_kwargs)
        fig.colorbar(im, ax=axes[1][i])
    plt.tight_layout()
    fig.savefig(save_name)
    plt.close()
    # err = plot_data['target_sst'] - plot_data['pred_sst']
