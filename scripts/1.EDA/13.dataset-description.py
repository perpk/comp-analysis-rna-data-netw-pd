import anndata as ad
import matplotlib.pyplot as plt

def main():
    genders = ["Male", "Female"]
    fontsize_title = 20
    fontsize = 16
    fontsize_legend = 12
    export_path = "/Volumes/Elements/aep/study/MSC/paper/figs/3.1_bars"

    ppmi_ad = ad.read_h5ad("/Volumes/Elements/PPMI_Project_133_RNASeq/ppmi_adata.h5ad")
    diag_mask = ppmi_ad.obs['Diagnosis'].isin(['PD', 'Control'])
    ppmi_ad_pd_ctrl = ppmi_ad[diag_mask]

    for gender in genders:
        ppmi_gender = ppmi_ad_pd_ctrl[ppmi_ad_pd_ctrl.obs['Gender'] == gender]

        age_counts_unique_patient = (
            ppmi_gender.obs
            .groupby(['Age_Group', 'Diagnosis'])['Patient']
            .nunique()
            .unstack(fill_value=0)
        )

        age_counts_samples = (
            ppmi_gender.obs
            .groupby(['Age_Group', 'Diagnosis'])['Sample']
            .nunique()
            .unstack(fill_value=0)
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f"Distribution of {gender} Patients and Samples per Cohort (PD, Control)", fontsize=fontsize_title)

        age_counts_unique_patient.plot(kind='bar', ax=ax1)
        ax1.set_title(f"{gender} - Unique Patients by Age Group",  fontsize=fontsize_title)
        ax1.set_ylabel("Number of Patients", fontsize=fontsize)
        ax1.set_xlabel("Age Group", fontsize=fontsize)
        ax1.legend(title="Diagnosis", fontsize=fontsize_legend)
        ax1.tick_params(axis='x', rotation=45, labelsize=fontsize)

        age_counts_samples.plot(kind='bar', ax=ax2)
        ax2.set_title(f"{gender} - Samples by Age Group", fontsize=fontsize_title)
        ax2.set_ylabel("Number of Samples", fontsize=fontsize)
        ax2.set_xlabel("Age Group", fontsize=fontsize)
        ax2.legend(title="Diagnosis", fontsize=fontsize_legend)
        ax2.tick_params(axis='x', rotation=45, labelsize=fontsize)

        plt.tight_layout()
        plt.savefig(f"{export_path}/distribution_age_groups_{gender}.png", dpi=600, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
