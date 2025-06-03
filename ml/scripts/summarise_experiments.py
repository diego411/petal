from pathlib import Path
import pandas as pd
import yaml
EXPERIMENTS_PATH = Path('ml/experiments')

if __name__ == '__main__':
    summary_df = pd.DataFrame(columns=[
        'experiment',
        'model',
        'version',
        'binary',
        'epoch',
        'validation_accuracy',
        'validation_auroc',
        'validation_macro_f1',
        'validation_weighted_f1',
        'validation_loss',
        'validation_precision',
        'validation_recall',
    ])
    i = 0

    for experiment in EXPERIMENTS_PATH.iterdir():
        if not experiment.is_dir():
            continue
        
        contents = list(experiment.iterdir())
        if len(contents) == 0:
            continue 

        versions = [dir for dir in contents if dir.is_dir() and dir.stem.startswith('version')]
        if len(versions) == 0:
            continue
        
        for version in versions:
            metrics_path = version / 'metrics.csv'
            if not metrics_path.exists():
                continue

            config_path = version / 'config.yaml'
            if not config_path.exists():
                continue
            
            with open(config_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            
            model_config = yaml_data['model']
            model_name = None
            architecture = None
            if 'init_args' in model_config:
                model_config = model_config['init_args']
            if 'pretrained_model_name' in model_config:
                model_name = model_config['pretrained_model_name']
            if 'class_path' in model_config:
                architecture = model_config['class_path']

            df = pd.read_csv(metrics_path)

            if not 'validation_macro_f1' in df:
                continue

            best_f1 = df['validation_macro_f1'].max()
            best_epoch = df[df['validation_macro_f1'] == best_f1].iloc[0]

            summary_df.loc[i] = pd.Series({
                'experiment': experiment.stem,
                'architecture': architecture,
                'pre_trained_model': model_name,
                'version': version.stem,
                'binary': model_config['n_output'] == 1,
                'epoch': best_epoch['epoch'],
                'validation_accuracy': best_epoch['validation_accuracy'].item(),
                'validation_auroc': best_epoch['validation_auroc'].item(),
                'validation_macro_f1': best_epoch['validation_macro_f1'].item(),
                'validation_weighted_f1': best_epoch['validation_weighted_f1'].item(),
                'validation_loss': best_epoch['validation_loss'].item(),
                'validation_precision': best_epoch['validation_precision'].item(),
                'validation_recall': best_epoch['validation_recall'].item(),
            })
                
            i = i + 1
    
    summary_df.sort_values(by=['validation_macro_f1'], ascending=False, inplace=True)
    summary_df.to_csv('ml/experiments/summary.csv', index=False)

