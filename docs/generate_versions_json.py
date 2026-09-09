import git
import json

repo = git.Repo('../.')


# GitHub has an upload limit of ~1GB for pages hence we limit the number of
# releases that we generate the docs for.
tags = list(reversed(sorted(repo.tags, key=lambda t:
                            t.commit.committed_datetime)))
if (len(tags) > 4):
    tags = tags[0:4]

def jsonobjectfunc(version):
    strversion = str(version)
    urlversion = "https://excalibur-neptune.github.io/NESO-Particles/" + strversion + "/sphinx/html/"
    return {"version": strversion, "url": urlversion}
    
json_contents = [jsonobjectfunc("main"), jsonobjectfunc("dev")]
for t in tags:
    tagobject = jsonobjectfunc(t)
    json_contents.append(tagobject)

with open('switcher.json', 'w') as fh:
    json.dump(json_contents, fh)
