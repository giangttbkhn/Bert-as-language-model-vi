import os
import re


def split_sentence_from_doc(path_to_text_file, path_save_file):
    sentences = []
    with open(path_to_text_file, 'r', encoding="utf-16le", errors='ignore') as f:
        contents = f.read()
        contents = re.sub('\.\.\.|', '', contents.strip())
        sentences = contents.split('.')
    if (sentences[-1] != ''):
        sentences.append('')
    i = 0
    while i < (len(sentences) - 2):
        tmp = sentences[i].strip()
        while (not sentences[i + 1].strip()[0].isupper()):
            tmp = tmp + sentences[i + 1]
            sentences[i] = ''
            i += 1
            if (i >= len(sentences) - 2):
                break
        sentences[i] = tmp
        i += 1
    i = 0
    with open(path_save_file, 'w', encoding="utf-8") as f:
        for sentence in sentences:
            if (sentence == ''):
                continue
            else:
                f.write(sentence.strip() + '.\n')
                i += 1
    return i

def batch_split_sentence_from_doc(folder_path, folder_save):
    if (not os.path.exists(folder_save)):
        try:
            os.makedirs(folder_save)
        except OSError:
            print("Creation of the directory %s failed" % folder_path)
    sentence = 0
    eror = 0
    file = 0
    for filename in os.listdir(folder_path):
        path_file = folder_path + filename
        path_save = folder_save + filename
        if (os.path.isfile(path_file) and filename.endswith(".txt")):
            try:
                sentence = sentence + split_sentence_from_doc(path_file, path_save)
                file +=1
            except:
                eror+=1


    print("toal file: ", file)
    print("toal sentence: ", sentence)
    print("eror: ", eror)


def agg_document(folder_path,file_path_save):
    with open(file_path_save, 'a+', encoding="utf-8") as total_file:
        for filename in os.listdir(folder_path):
            path_file = folder_path + filename
            if (os.path.isfile(path_file) and filename.endswith(".txt")):
                with open(path_file, 'r', encoding="utf-8") as f:
                    content = f.read()
                    total_file.write(content)
                    total_file.write('\n')

folder_root_path = '/Users/trinhgiang/Downloads/VNTC-master/Data/10Topics/Ver1.1/Train_Full/Vi tinh/'
folder_save_path = '/Users/trinhgiang/Downloads/Sentence_output/Vi_tinh/'
file_agg_save = '/Users/trinhgiang/Downloads/Sentence_output/total/Vi_tinh.txt'
batch_split_sentence_from_doc(folder_root_path,folder_save_path)
agg_document(folder_save_path,file_agg_save)