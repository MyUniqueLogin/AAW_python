import pyopencl as cl

print('Wersja PyOpenCL: ' + cl.VERSION_TEXT)
print('Wersja OpenCL: ' + '.'.join(map(str, cl.get_cl_header_version())) + '\n')

print('Platformy:')
platforms = cl.get_platforms()

for plat in platforms:  
    print('{} ({})'.format(plat.name, plat.vendor))
    print('\tVersion: ' + plat.version)
    print('\tProfile: ' + plat.profile)

    devices = plat.get_devices(cl.device_type.ALL)

    print('\tAvailable devices: ')
    
    for dev in devices:
        indent = '\t\t'
        print(indent + '{} ({})'.format(dev.name, dev.vendor))

        indent = '\t\t\t'
        flags = [('Version', dev.version),
                 ('Type', cl.device_type.to_string(dev.type)),
                 ('Memory (global)', str(dev.global_mem_size)),
                 ('Memory (local)', str(dev.local_mem_size)),
                 ('Address bits', str(dev.address_bits)),
                 ('Max work item dims', str(dev.max_work_item_dimensions)),
                 ('Max work group size', str(dev.max_work_group_size)),
                 ('Max compute units', str(dev.max_compute_units)),
                 ('Driver version', dev.driver_version),
                 ('Image support', str(bool(dev.image_support))),
                 ('Little endian', str(bool(dev.endian_little))),
                 ('Device available', str(bool(dev.available))),
                 ('Compiler available', str(bool(dev.compiler_available)))]

        [print(indent + '{0:<25}{1:<10}'.format(name + ':', flag)) for name, flag in flags]
    print('')
