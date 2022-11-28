'''
Load private info from files
'''
import json

subject_file = '/Fridge/users/julia/project_decoding_jip_janneke/data/subjects.json'
annot_file = '/Fridge/users/julia/project_decoding_jip_janneke/data/annot.json'

def get_subject_by_code(code):
    subjects = json.load(open(subject_file, 'r'))
    return subjects[code]

def get_annot_by_code(subject):
    annots = json.load(open(annot_file, 'r'))
    return annots[subject]

def get_subjects():
    return json.load(open(subject_file, 'r'))

def get_subjects_by_code(code_list):
    subjects = json.load(open(subject_file, 'r'))
    return [subjects[c] for c in code_list]

def get_annots_by_code(subject_list):
    annots = json.load(open(annot_file, 'r'))
    return [annots[s] for s in subject_list]