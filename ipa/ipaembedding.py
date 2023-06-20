import warnings
import ipapy
from ipapy.ipachar import *

import numpy as np

import argparse
import logging
import warnings
from collections import defaultdict

import pandas as pd


class IPAEmbedding(object):

    def __init__(self):

        descriptors = {
            "global" : { # TODO: Derived global features could be interesting,
                        # i.e. a label <nasal> for both nasal consonants and nazalised vowels,
                        # or <front> for both front vowels and consonants
                "type": [d.canonical_label for d in DG_TYPES.descriptors]
            },

            "consonant" : {
                "voicing": [d.canonical_label for d in DG_C_VOICING.descriptors],
                "place"  : [d.canonical_label for d in DG_C_PLACE.descriptors],
                "manner" : [d.canonical_label for d in DG_C_MANNER.descriptors],
            },

            "vowel" : {
                "height"   : [d.canonical_label for d in DG_V_HEIGHT.descriptors],
                "backness" : [d.canonical_label for d in DG_V_BACKNESS.descriptors],
                "roundness": [d.canonical_label for d in DG_V_ROUNDNESS.descriptors],
            },

            "diacritic" : {
                "feature": [d.canonical_label for d in DG_DIACRITICS.descriptors],
            },

            "suprasegmental" : {
                "stress": [d.canonical_label for d in DG_S_STRESS.descriptors],
                "length": [d.canonical_label for d in DG_S_LENGTH.descriptors],
                "break" : [d.canonical_label for d in DG_S_BREAK.descriptors],
            },

            "tone" : {
                "level"  : [d.canonical_label for d in DG_T_LEVEL.descriptors],
                "contour": [d.canonical_label for d in DG_T_CONTOUR.descriptors],
                "global" : [d.canonical_label for d in DG_T_GLOBAL.descriptors],
            },

        }

        features = {}

        for d1, values in descriptors.items():
            for d2, labels in values.items():
                for label in labels:
                    if label in features:
                        warnings.warn("Label ({}:{}) already in features as <{}>".format(label, (d1, d2), features[label]))

                    features[label] = {
                        "descriptor_major": d1,
                        "descriptor_minor": d2,
                        "id": "_".join([d1, d2, label])
                    }
                    
        self.features = features
        self.feature_names = [features[f]["id"] for f in self.features]
    
    def __getitem__(self, s):
        description = self.get_description(s)
        combined = False
        if not description and len(s)>1:     # TODO: This is a naive way of combining IPA symbols.
                                             # e.g., we should only allow one global type (i.e. <consonant> should overrule <diacritic>)
            description = []
            for symbol in list(s):
                description += self.get_description(symbol)
            
            combined=True
            logging.debug("The feature set of '{}' is a combination of features from the symbols {}".format(s, list(s)))
        
        description = set(description)

        return [1 if feature in description else 0 for feature in self.features], combined
    
    def get_description(self, s):
        try:
            return ipapy.UNICODE_TO_IPA[s].canonical_representation.split()
        except KeyError:
            logging.debug("Could not retrieve an IPA reprentation of the segment '{}'".format(s))
            return []