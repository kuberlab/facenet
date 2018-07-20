
def parse_check_ouput(out):
    result = {}
    for l in out.split('\n'):
        l = l.strip()
        p = l.split(':')
        if len(p)>0:
            if p[0].startswith('Obtained '):
                name = p[0][len('Obtained '):].strip()
                value = p[1].strip().rstrip(',')
                result[name] = value
