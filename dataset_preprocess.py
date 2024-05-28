import os
import copy

def is_similar(s1, s2):
    count = 0.0
    for token in s1.split(' '):
        if token in s2:
            count += 1
    if count / len(s1.split(' ')) >= 0.9 and count / len(s2.split(' ')) >= 0.9:
        return True
    else:
        return False

def assemble_aspects(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('$ t $','$T$').strip()

    def unify_same_samples(same_samples):
        text = same_samples[0][0].replace('$T$', same_samples[0][1])
        polarities = [-1]*len(text.split())
        tags=['O']*len(text.split())
        samples = []
        for sample in same_samples:
            polarities_tmp = copy.deepcopy(polarities)

            try:
                asp_begin = (sample[0].split().index('$T$'))
                asp_end = sample[0].split().index('$T$')+len(sample[1].split())
                for i in range(asp_begin, asp_end):
                    polarities_tmp[i] = int(sample[2])+1
                    if i - sample[0].split().index('$T$')<1:
                        tags[i] = 'B-ASP'
                    else:
                        tags[i] = 'I-ASP'
                samples.append([text, tags, polarities_tmp])
            except:
                print(sample[0])

        return samples

    samples = []
    aspects_in_one_sentence = []
    for i in range(0, len(lines), 3):

        if len(aspects_in_one_sentence) == 0:
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
            continue
        if is_similar(aspects_in_one_sentence[-1][0], lines[i]):
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
        else:
            samples.extend(unify_same_samples(aspects_in_one_sentence))
            aspects_in_one_sentence = []
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])

    return samples

def refactor_dataset(fname, dist_fname):
    lines = []
    samples = assemble_aspects(fname)

    for sample in samples:
        for token_index in range(len(sample[1])):
            token, label, polarty = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))

        lines.append('\n')
    if os.path.exists(dist_fname):
        os.remove(dist_fname)
    fout = open(dist_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()

if __name__ == "__main__":
    # Process only restaurant dataset
    refactor_dataset(
        r"../datasets_origin/semeval14/Restaurants_Train.xml.seg",
        r"../atepc_datasets/restaurant/Restaurants.atepc.train.dat",
    )
    refactor_dataset(
        r"../datasets_origin/semeval14/Restaurants_Test_Gold.xml.seg",
        r"../atepc_datasets/restaurant/Restaurants.atepc.test.dat",
    )
