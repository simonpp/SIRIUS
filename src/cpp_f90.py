import sys

# @fortran begin function                                    {type} {name} {doc-string}
# @fortran       argument {in|out|inout} {required|optional} {type} {name} {doc-string}
# @fortran end

in_type_map = {
    'void*'   : 'type(C_PTR)',
    'int'     : 'integer(C_INT)',
    'double'  : 'real(C_DOUBLE)',
    'string'  : 'character(C_CHAR)',
    'bool'    : 'logical(C_BOOL)',
    'complex' : 'complex(C_DOUBLE)'
}

def write_str_to_f90(o, string):
    n = 80
    while len(string) > n:
        o.write(string[:n] + '&\n&')
        string = string[n:]
    o.write(string)
    o.write('\n')

def write_function(o, func_name, func_suffix, func_type, func_args, func_doc):
    o.write('!> @brief ' + func_doc + '\n')
    for a in func_args:
        o.write('!> @param [' + a['intent'] + '] ' + a['name'] + ' ' + a['doc'] + '\n')

    if func_type == 'void':
        string = 'subroutine '
    else:
        string = 'function '
    string = string + func_name + func_suffix + '('
    va = [a['name'] for a in func_args]
    string = string + ','.join(va)
    string = string + ')'
    if func_type != 'void':
        string = string + ' result(res)'
    write_str_to_f90(o, string)
    o.write('implicit none\n')

    for a in func_args:
        o.write(in_type_map[a['type']])
        if not a['required']:
            o.write(', optional, target')
        if a['type'] == 'string':
            o.write(', dimension(*)')
        o.write(', intent(' + a['intent'] + ') :: ' + a['name'])
        o.write('\n')

    if func_type != 'void':
        o.write(in_type_map[func_type] + ' :: res\n')

    for a in func_args:
        if not a['required']:
            o.write('type(C_PTR) :: ' + a['name'] + '_ptr = C_NULL_PTR\n')
    o.write('interface\n')

    if func_type == 'void':
        string = 'subroutine '
    else:
        string = 'function '
    string = string + func_name + '_aux('
    va = [a['name'] for a in func_args]
    string = string + ','.join(va)
    string = string + (')')
    if (func_type == 'void'):
        string = string + '&'
    else:
        string = string + ' result(res)&'
    write_str_to_f90(o, string)
    o.write('&bind(C, name="'+func_name+'")\n')

    o.write('use, intrinsic :: ISO_C_BINDING\n')
    for a in func_args:
        if not a['required']:
            o.write('type(C_PTR)')
            o.write(', value')
        else:
            o.write(in_type_map[a['type']])
            if a['type'] == 'string':
                o.write(', dimension(*)')
        o.write(', intent(' + a['intent'] + ') :: ' + a['name'])
        o.write('\n')

    if func_type != 'void':
        o.write(in_type_map[func_type] + ' :: res\n')

    if func_type == 'void':
        o.write('end subroutine\n')
    else:
        o.write('end function\n')
    o.write('end interface\n')

    for a in func_args:
        if not a['required']:
            o.write('if (present('+a['name']+')) ' + a['name'] + '_ptr = C_LOC(' + a['name'] + ')\n')

    if (func_type == 'void'):
        string = 'call '
    else:
        string = 'res = '
    string = string + func_name + '_aux('
    va = []
    for a in func_args:
        if not a['required']:
            va.append(a['name'] + '_ptr')
        else:
            va.append(a['name'])
    string = string + ','.join(va)
    string = string + ')'
    write_str_to_f90(o, string)

    if func_type == 'void':
        o.write('end subroutine ')
    else:
        o.write('end function ')
    o.write(func_name + func_suffix + '\n\n')

def main():
    f = open(sys.argv[1], 'r') 
    o = open('generated.f90', 'w')
    o.write('! Warning! This file is automatically generated using cpp_f90.py script!\n\n')
    o.write('!> @file generated.f90\n')
    o.write('!! @brierf Autogenerated interface to Fortran.\n')

    o.write('!\n')
    while (True):
        line = f.readline()
        if not line: break

        i = line.find('@fortran')
        if i > 0:
            v = line[i:].split()
            if v[1] == 'begin' and v[2] == 'function':
                func_type = v[3]
                func_name = v[4]
                if (v[5][0] == '_'):
                    func_suffix = v[5]
                    func_doc = ' '.join(v[6:])
                else:
                    func_suffix = ''
                    func_doc = ' '.join(v[5:])

                func_args = []

                while (True):
                    line = f.readline()

                    i = line.find('@fortran')
                    if i > 0:
                        v = line[i:].split()
                        if v[1] == 'argument':
                            if v[3] == 'required':
                                arg_required = True
                            else:
                                arg_required = False
                            arg_doc = ' '.join(v[6:])
                            func_args.append({'type'     : v[4],
                                              'intent'   : v[2],
                                              'required' : arg_required, 
                                              'name'     : v[5],
                                              'doc'      : arg_doc})
                        if v[1] == 'end': break

            if v[1] == 'end':
                write_function(o, func_name, func_suffix, func_type, func_args, func_doc)


    f.close()
    o.close()

if __name__ == "__main__":
    main()
