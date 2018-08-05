import os, subprocess

bids_dir = '/data1/users/adoyle/IBIS/BIDS/'


if __name__ == '__main__':
    print('hack to get mriqc to work')
    for filename in os.listdir(bids_dir):
        subj_id = filename.split('-')[-1]

        print('Starting participant:', subj_id)
        subprocess.run(['docker', 'run', '-it', '-v', '/data1/users/adoyle/IBIS/BIDS/:/data:ro', '-v', '/data1/users/adoyle/IBIS/mriqc_output/:/out poldracklab/mriqc:latest', '/data', '/out', 'participant', '--participant-label', subj_id, '--no-sub'], shell=True, check=True)