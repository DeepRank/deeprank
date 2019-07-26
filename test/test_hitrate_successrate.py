import unittest
from deeprank.utils.plot_utils import *
import os
import pandas as pd
from time import time


"""
Some requirement of the naming of the files:
    1. case ID canNOT have underscore '_', e.g., '1ACB_CD'
    2. decoy file name should have this format: 2w83-AB_20.pdb (caseID_xxx.pdb)
    3. pssm file name should have this format: 2w83-AB.A.pssm (caseID.chainID.pssm or caseID.chainID.pdb.pssm)
"""


class TestGenerateData(unittest.TestCase):
    """Test the calculation of hit rate and success rate."""

    rawScoreFL = 'hitrate_successrate/scores_raw.tsv'
    groundTruth_FL = 'hitrate_successrate/success_hitrate_ANS.tsv'



    def test_1_hitrate_success_averaged_over_cases(self):

        def compare_hitrate_success_one_case(expected_df, real_df, caseID):

            expected_df = expected_df.reset_index()
            real_df = real_df.reset_index()

            columns = ['success_DR', 'hitRate_DR', 'success_HS', 'hitRate_HS']

            for col in columns:
                expected = expected_df[col]
                real = real_df[col]
                error_msg = f"{col} for {label} {caseID} is not correct!"
                assert (expected == real).all() , error_msg


        # calculate hitrate and success
        rawScore = pd.read_csv(self.rawScoreFL, sep = '\t')
        hitrate_success_df = evaluate(rawScore)

        # compare with the grount truth
        groundTruth_df = pd.read_csv(self.groundTruth_FL, sep = '\t')

        labels = groundTruth_df['label'].unique()
        caseIDs ={}
        truth_grp = groundTruth_df.groupby('label')

        for label, df in truth_grp:
            caseIDs[label] = df['caseID'].unique()

        for label in labels:
            for caseID in caseIDs[label]:
                idx = (groundTruth_df['label'] == label ) & (groundTruth_df['caseID'] == caseID)
                expected_df = groundTruth_df[idx]

                idx = (hitrate_success_df['label'] == label ) & (hitrate_success_df['caseID'] == caseID)
                real_df = hitrate_success_df[idx]
                compare_hitrate_success_one_case(expected_df, real_df, caseID)


if __name__ == "__main__":
    unittest.main()


