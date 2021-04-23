import filecmp
import pathlib
import sys

if len(sys.argv) < 2:
    print("usage: {} <compare filename>".format(sys.argv[0]))
    exit(1)

src_path = pathlib.Path(sys.argv[1])
p = pathlib.Path('/home/yuchialiu/workspace/TensorStore')
ls = list(p.glob('**/*/{}'.format(sys.argv[1].split('/')[-1])))
print('source file is {}'.format(src_path.resolve()))
result = True

if len(ls) <= 0:
    print("doesn't get any files to compare")
    exit(1)

print('comparing {} files'.format(len(ls)))
for path in ls:
    # print(str(path.resolve()))
    if filecmp.cmp(str(src_path.resolve()), str(path.resolve())) is not True:
        print('{} is not the same'.format(path))
        result = result and False

if result is not True:
    print('someone is not the same')
else:
    print('all clear')