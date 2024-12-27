project(
    'libquspin',
    'cpp',
    version: '0.1.0',
    default_options: ['warning_level=3', 'cpp_std=c++20'],
)

if host_machine.system() == 'windows'
    add_project_arguments('/bigobj', language: 'cpp')
elif host_machine.system() == 'darwin'
    add_project_arguments(
        '-fopenmp',
        '-mmacosx-version-min=11.1',
        language: 'cpp',
    )
    add_project_link_arguments(
        '-flto',
        '-fopenmp',
        '-mmacosx-version-min=11.1',
        language: 'cpp',
    )
elif host_machine.system() == 'linux'
    add_project_arguments('-fopenmp', language: 'cpp')
    add_project_link_arguments('-fopenmp', language: 'cpp')
endif

add_global_arguments(
    '-DVERSION_INFO=' + meson.project_version(),
    language: 'cpp',
)

include = include_directories('include')

dependencies = []

src_files = files(
    'src' / 'array' / 'array.cpp',
    'src' / 'array' / 'kernels.cpp',
    'src' / 'scalar' / 'scalar.cpp',
    'src' / 'scalar' / 'reference.cpp',
    'src' / 'dtype' / 'dtype.cpp',
    'src' / 'qmatrix' / 'kernel' / 'sum.cpp',
    'src' / 'qmatrix' / 'qmatrix.cpp',
)

if host_machine.system() == 'windows'
    libquspin = static_library(
        'quspin',
        src_files,
        include_directories: include,
        name_suffix: 'lib',
        dependencies: dependencies,
    )
elif host_machine.system() == 'darwin'

    libquspin = shared_library(
        'quspin',
        src_files,
        include_directories: include,
        dependencies: dependencies,
    )
else # linux
    libquspin = shared_library(
        'quspin',
        src_files,
        include_directories: include,
        dependencies: dependenciesxs,

    )
endif

libquspin_dep = declare_dependency(
    include_directories: include,
    link_with: libquspin,
)

dependencies += libquspin_dep


test_files = [
    ('test' / 'array' / 'iterator.cpp'),
    ('test' / 'array' / 'constructor.cpp'),
    ('test' / 'array' / 'kernels.cpp'),
    ('test' / 'dtype.cpp'),
    ('test' / 'detail' / 'select.cpp'),
    # ('test' / 'basis_scratch.cpp'),
    ('test' / 'qmatrix_sketch.cpp'),
    ('test' / 'scratch.cpp')
]

foreach test_file : test_files
    test_name = test_file.strip('.cpp').replace('/', '_').replace('\\', '_')
    exe = executable(test_name, test_file, dependencies: dependencies)
    test(test_name, exe)
endforeach
