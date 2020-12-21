infile = open('solv_param.txt', 'r')
outfile = open('solv_param_if.py', 'w')

#if atom.name() == 'BB' and/or atom.resn == 'ALA':
#    atom.solv() = float(row[3])

for i, line in enumerate(infile):
    row = line.split(' ')
    if '' in row:
        row.remove('')
    print(row)
    if i == 0:
        outfile.write('if atom.name() == "%s" or atom.name().startswith("%s"):\n' %(row[5], row[8][:2]))
        outfile.write('    atom.solv() = %s\n' %row[3].replace(')', ''))
    else:
        if '*' in row[5]:
            outfile.write('elif atom.name().startswith("%s") and atom.resn()=="%s":\n' %(row[5].replace('*',''), row[8][:3]))
            outfile.write('    atom.solv() = %s\n' %row[3].replace(')', ''))
        else:
            outfile.write('elif atom.name() == "%s" and atom.resn()=="%s":\n' %(row[5].replace('*',''), row[8][:3]))
            outfile.write('    atom.solv() = %s\n' %row[3].replace(')', ''))

infile.close()
outfile.close()
