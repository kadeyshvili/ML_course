from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('&', '&amp;')
    with open(filename, 'w') as file:
        file.write(filedata)

    tree = ET.parse(filename)
    root = tree.getroot()
    labeled_aligments = []
    sentence_pairs = []
    for elem in root:
        english_sentence = elem.find('english').text.split()
        czech_sentence = elem.find('czech').text.split()
        sentence_pairs.append(SentencePair(english_sentence, czech_sentence))

        sure = []
        trying = elem.find('sure')
        if trying is not None:
            if trying.text is not None:
                for align in elem.find('sure').text.split():
                    source_pos, target_pos = map(int, align.split('-'))
                    sure.append((source_pos,target_pos))

        possible = []
        trying = elem.find('possible')
        if trying is not None:
            if trying.text is not None:
                for align in elem.find('possible').text.split():
                    source_pos, target_pos = map(int, align.split('-'))
                    possible.append((source_pos, target_pos))

        labeled_aligments.append(LabeledAlignment(sure, possible))    
        
    return sentence_pairs, labeled_aligments



def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_dict = {}
    target_dict = {}
    src_idx = 0
    tgt_idx = 0

    for pair in sentence_pairs:
        for src_word in pair.source:
            if src_word not in source_dict:
                source_dict[src_word] = src_idx
                src_idx += 1
        for tgt_word in pair.target:
            if tgt_word not in target_dict:
                target_dict[tgt_word] = tgt_idx
                tgt_idx += 1
    if freq_cutoff is None:
        return (source_dict, target_dict)
    
    src_dict = dict(sorted(source_dict.items(), key=lambda x:x[1], reverse=True)[:freq_cutoff])
    tgt_dict = dict(sorted(target_dict.items(), key=lambda x:x[1], reverse=True)[:freq_cutoff])
       
    return  src_dict, tgt_dict




def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    res = []
    for sentence in sentence_pairs:
        src = []
        tgt = []
        none_in_src = False
        none_in_tgt = False
        for src_word in sentence.source:
            if src_word in source_dict:
                src.append(source_dict[src_word])
            else:
                none_in_src = True
            
        for tgt_word in sentence.target:
            if tgt_word in target_dict:
                tgt.append(target_dict[tgt_word])
            else:
                none_in_tgt = True
            
        if none_in_src != True and  none_in_tgt != True:
            res.append(TokenizedSentencePair(src, tgt))
    return res


