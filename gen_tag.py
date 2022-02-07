import os,os.path
import shutil

tags_map = {}
archive_yaer = {}

def file_stat(rel_path, file_name):
    print('parsing ' + rel_path + ' ...')
    file_year = file_name.split('-')[0]
    archive_yaer[file_year] = 1
    ff = open(rel_path, 'r')
    start = False
    for line in ff:
        ll = line.strip()
        if ll == '---':
            if start == False:
                start = True
                continue
            else:
                break
        if start:
            if ll[0:5] == 'tags:':
                tag_list = ll[5:].strip().split(' ')
                for tag in tag_list:
                    if len(tag) > 0:
                        tags_map[tag] = 1
    ff.close()

# main process start

# find all files under _posts ...
for r,d,f in os.walk('./_posts/'):
    for file_name in f:
        rel_path = os.path.join(r, file_name)
        # rel_path = 相对路径文件名 file_name = 文件名
        file_stat(rel_path, file_name)

# generate tags/*.html
tag_pattern = '---\n'
tag_pattern += 'layout: tag_page\n'
tag_pattern += 'title: %s\n'
tag_pattern += 'tag_name: %s\n'
tag_pattern += 'permalink: /tags/%s/\n'
tag_pattern += '---\n'

try:
    shutil.rmtree('tags')
except FileNotFoundError:
    pass
os.mkdir('tags')

print('tags found : ' + ' '.join(tags_map.keys()))

for tag in tags_map.keys():
    f = open('tags/' + tag + '.html', 'w')
    f.write(tag_pattern % (tag, tag, tag))
    f.close()