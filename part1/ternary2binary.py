TRAINING_IMGS = "digitdata/trainingimages"
TEST_IMGS = "digitdata/testimages"

def t2b():
    files = TRAINING_IMGS, TEST_IMGS
    for file in files:
        f = open(file)
        content = f.read()
        content = content.replace('+', '#')
        wf = open(file  + '_bin', "w")
        wf.write(content)

if __name__ == '__main__':
    t2b()
