import sys
START_N=6501
END_N=8000

LANG_PAIRS=['bn-en', 'en-bn', 'bs-en', 'en-bs', 'et-en', 'en-et', 'fa-en',
        'en-fa', 'id-en', 'en-id', 'mk-en', 'en-mk', 'ms-en', 'en-ms', 'en-ta',
        'ta-en', 'vi-en', 'en-vi', 'zh-en', 'en-zh', 'en-de', 'de-en', 'en-it',
        'it-en', 'en-es', 'es-en', 'en-ru', 'ru-en', 'en-fr',
        'fr-en', 'ja-en', 'en-ja', 'es-de', 'de-es', 'it-fr', 'fr-it', 'es-pt',
        'pt-es', 'de-pt', 'pt-de']

def make_dev(infile, outfile):
    with open(outfile, 'w') as o:
        uniq_wdcount = 0
        last_word = None
        for line in open(infile, 'r'):
            srcwd, trgwd = line.split()
            if srcwd == last_word and uniq_wdcount >= START_N:
                o.write(line)
            elif srcwd != last_word:
                last_word = srcwd
                uniq_wdcount += 1
                if uniq_wdcount > END_N:
                    print('Dev creation complete. Exiting.')
                    o.close()
                    return
                elif uniq_wdcount >= START_N:
                    o.write(line)

for lang_pair in LANG_PAIRS:
    infile = lang_pair + '/' + lang_pair + '.txt'
    outfile = (lang_pair + '/dev/' + lang_pair + '.' + str(START_N) + '-' +
            str(END_N) + '.txt')
    make_dev(infile, outfile)
