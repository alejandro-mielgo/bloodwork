import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


blood_types:dict[str:float] = {'AB+':0.03,'AB-':0.01,'A+':0.31,'A-':0.07,'B+':0.095,'B-':0.025,'O+':0.38,'O-':0.08}
blood_types_acc:dict[str:float] = {}

acc:float = 0
for blood_type,percentage in blood_types.items():
    acc = acc + percentage
    blood_types_acc[blood_type] = acc


compatible:dict[str:dict[str:float]] = {}
compatible['AB+'] = {'O-':True, 'O+':True, 'B-':True, 'B+':True,  'A-':True,  'A+':True,  'AB-':True, 'AB+':True}
compatible['AB-'] = {'O-':True, 'O+':False,'B-':True, 'B+':False, 'A-':True,  'A+':False, 'AB-':True, 'AB+':False}
compatible['A+'] =  {'O-':True, 'O+':True, 'B-':False,'B+':False, 'A-':True,  'A+':True,  'AB-':False, 'AB+':False}
compatible['A-'] =  {'O-':True, 'O+':False,'B-':False,'B+':False, 'A-':True,  'A+':False, 'AB-':False, 'AB+':False}
compatible['B+'] =  {'O-':True, 'O+':True, 'B-':True, 'B+':True,  'A-':False, 'A+':False, 'AB-':False, 'AB+':False}
compatible['B-'] =  {'O-':True,'O+':False, 'B-':True, 'B+':False, 'A-':False, 'A+':False, 'AB-':False, 'AB+':False}
compatible['O+'] =  {'O-':True,'O+':True,  'B-':False,'B+':False, 'A-':False, 'A+':False, 'AB-':False, 'AB+':False}
compatible['O-'] =  {'O-':True,'O+':False, 'B-':False,'B+':False, 'A-':False, 'A+':False, 'AB-':False, 'AB+':False}


def assign_patient_blood_type(blood_types:dict[str:float])->str:
    rand = np.random.rand()
    for blood_type,acc_percentage in blood_types.items():
        if rand <= acc_percentage:
            return blood_type
        

def assign_donor_blood_type(blood_types:dict[str:float],recipient_blood_type:str)->str:
    compatible_blood_types:list[str] = []
    for key in compatible[recipient_blood_type]:
        if compatible[recipient_blood_type][key]:
            compatible_blood_types.append(key)

    acc_blood_types:dict[str:float] = {}
    acc:float = 0
    for blood_type in compatible_blood_types:
        acc = acc + blood_types[blood_type]
        acc_blood_types[blood_type] = acc

    
    rand = np.random.rand()*acc
    # print(compatible_blood_types)
    # print(acc_blood_types)
    # print(rand)
    for blood_type,acc_percentage in acc_blood_types.items():
        if rand <= acc_percentage:
            return blood_type
        

def run_simulation(n_samples:int, indiviual_error_rate:float):

    sim:dict[str:list] = {}
    sim['index'] = list(range(n_samples))

    sim['receptor_true'] = [assign_patient_blood_type(blood_types_acc) for _ in range(n_samples)]
    sim['error_receptor_label'] = [False if np.random.rand() > indiviual_error_rate else True for _ in range(n_samples)]
    sim['receptor_label'] = [sim['receptor_true'][i] if sim['error_receptor_label'][i]==False else assign_patient_blood_type(blood_types_acc) for i in range(n_samples)]
    
    sim['donor_label'] = [assign_donor_blood_type(blood_types,sim['receptor_label'][i]) for i in range(n_samples)]
    sim['error_donor_label'] = [False if np.random.rand() > indiviual_error_rate else True for _ in range(n_samples)]
    sim['donor_true'] = [sim['donor_label'][i] if sim['error_donor_label'][i]==False else assign_patient_blood_type(blood_types_acc) for i in range(n_samples)]

    sim['compatible'] = [compatible[sim['receptor_true'][i]][sim['donor_true'][i]] for i in range(n_samples)]

    df = pd.DataFrame(sim)


    df['issue'] = ~df['compatible']
    df['error'] = df['error_donor_label'] | df['error_receptor_label']
    return df


def generate_summary(df:pd.DataFrame) -> pd.DataFrame:
    receptors = df.groupby('receptor_true')['issue'].count().rename('receptors')
    donors = df.groupby('donor_true')['issue'].count().rename('donnors')
    labeling_error= df.groupby('receptor_true')['error'].sum().rename('labeling_error')
    bad_outcomes = df.groupby('receptor_true')['issue'].sum().rename('bad_outcomes')
    summary = pd.concat([receptors,donors,labeling_error,bad_outcomes,],axis=1)
    summary['vulnerability'] = round(summary['bad_outcomes']/summary['labeling_error'],2)

    return summary


def plot_outcomes(summary:pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot total counts on the left y-axis
    summary['receptors'].plot(kind='bar', ax=ax1, color='cornflowerblue', position=0, width=0.4, label='Total Patients' )
    ax1.set_ylabel('Total Patients', color='black')
    ax1.tick_params(axis='y', labelcolor='cornflowerblue')
    ax1.set_xlabel('Blood Type')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot bad outcomes on the right y-axis
    summary['bad_outcomes'].plot(kind='bar', ax=ax2, color='tomato', position=1, width=0.4, label='Bad Outcomes')
    ax2.set_ylabel('Bad Outcomes', color='black')
    ax2.tick_params(axis='y', labelcolor='tomato')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Total Patients vs. Bad Outcomes per Blood Type')
    plt.margins(x=25)
    plt.subplots_adjust(right=0.95)
    ax1.set_xlim(-0.7, 7.7)

    return fig


def get_simulation_stats(df:pd.DataFrame, n_samples:int)->dict:
    donor_wrong_label:float = df['error_donor_label'].sum()
    receptor_wrong_label:float = df['error_receptor_label'].sum()
    bad_outcomes = df['issue'].sum()
    return {"donor_wrong_label":donor_wrong_label,
            "receptor_wrong_label":receptor_wrong_label,
            "bad_outcomes":bad_outcomes,
            "n_samples":n_samples}


def plot_vulnerability(summary:pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.bar(summary.index, summary['vulnerability'],color='tomato')
    plt.xlabel('Blood Type')
    plt.ylabel('Bad Outcome Rate if there is a labeling error')
    plt.title('Vulnerability to labeling error per blood type')
    return fig