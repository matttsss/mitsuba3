#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/jit.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/python/python.h>


// core
MI_PY_DECLARE(filesystem);
MI_PY_DECLARE(Object);
MI_PY_DECLARE(Cast);
MI_PY_DECLARE(Struct);
MI_PY_DECLARE(Appender);
MI_PY_DECLARE(ArgParser);
MI_PY_DECLARE(Bitmap);
MI_PY_DECLARE(Formatter);
MI_PY_DECLARE(FileResolver);
MI_PY_DECLARE(Logger);
MI_PY_DECLARE(MemoryMappedFile);
MI_PY_DECLARE(Stream);
MI_PY_DECLARE(DummyStream);
MI_PY_DECLARE(FileStream);
MI_PY_DECLARE(MemoryStream);
MI_PY_DECLARE(ZStream);
MI_PY_DECLARE(ProgressReporter);
MI_PY_DECLARE(rfilter);
MI_PY_DECLARE(Thread);
MI_PY_DECLARE(Timer);
MI_PY_DECLARE(Properties);
MI_PY_DECLARE(parser);
MI_PY_DECLARE(misc);

// render
MI_PY_DECLARE(BSDFContext);
MI_PY_DECLARE(EmitterExtras);
MI_PY_DECLARE(RayFlags);
MI_PY_DECLARE(MicrofacetType);
MI_PY_DECLARE(PhaseFunctionExtras);
MI_PY_DECLARE(Spiral);
MI_PY_DECLARE(Sensor);
MI_PY_DECLARE(VolumeGrid);
MI_PY_DECLARE(FilmFlags);
MI_PY_DECLARE(DiscontinuityFlags);

NB_MODULE(mitsuba_ext, m) {
    // Temporarily change the module name (for pydoc)
    m.attr("__name__") = "mitsuba";

    // Expose some constants in the main `mitsuba` module
    m.attr("__version__")      = MI_VERSION;
    m.attr("MI_VERSION")       = MI_VERSION;
    m.attr("MI_VERSION_MAJOR") = MI_VERSION_MAJOR;
    m.attr("MI_VERSION_MINOR") = MI_VERSION_MINOR;
    m.attr("MI_VERSION_PATCH") = MI_VERSION_PATCH;
    m.attr("MI_YEAR")          = MI_YEAR;
    m.attr("MI_AUTHORS")       = MI_AUTHORS;

#if defined(NDEBUG)
    m.attr("DEBUG") = false;
#else
    m.attr("DEBUG") = true;
#endif

#if defined(MI_ENABLE_CUDA)
    m.attr("MI_ENABLE_CUDA") = true;
#else
    m.attr("MI_ENABLE_CUDA") = false;
#endif

#if defined(MI_ENABLE_EMBREE)
    m.attr("MI_ENABLE_EMBREE") = true;
#else
    m.attr("MI_ENABLE_EMBREE") = false;
#endif

    // Initialize reference counting hooks for mitsuba::Object
    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            if (!nb::is_alive())
                return;

            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        }
    );

    m.def("set_log_level", [](mitsuba::LogLevel level) {
        if (!mitsuba::logger()) {
            Throw("No Logger instance is set on the current thread! This is likely due to "
                  "set_log_level being called from a non-Mitsuba thread. You can manually set a "
                  "thread's ThreadEnvironment (which includes the logger) using "
                  "ScopedSetThreadEnvironment e.g.\n"
                  "# Main thread\n"
                  "env = mi.ThreadEnvironment()\n"
                  "# Secondary thread\n"
                  "with mi.ScopedSetThreadEnvironment(env):\n"
                  "   mi.set_log_level(mi.LogLevel.Info)\n"
                  "   mi.Log(mi.LogLevel.Info, 'Message')\n");
        }

        mitsuba::logger()->set_log_level(level);
    }, "Sets the log level.");
    m.def("log_level", []() {
        return mitsuba::logger()->log_level();
    }, "Returns the current log level.");

    Jit::static_initialization();
    Thread::static_initialization();
    Logger::static_initialization();
    Bitmap::static_initialization();
    Profiler::static_initialization();

#if defined(NDEBUG)
    // Default log level in Python should be Warn (unless we compiled in debug)
    mitsuba::logger()->set_log_level(mitsuba::LogLevel::Warn);
#endif

    // Append the mitsuba directory to the FileResolver search path list
    ref<FileResolver> fr = mitsuba::file_resolver();
    fs::path base_path = util::library_path().parent_path();
    if (!fr->contains(base_path))
        fr->append(base_path);

    // Register python modules
    MI_PY_IMPORT(filesystem);
    MI_PY_IMPORT(Object);
    MI_PY_IMPORT(Cast);
    MI_PY_IMPORT(Struct);
    MI_PY_IMPORT(Appender);
    MI_PY_IMPORT(ArgParser);
    MI_PY_IMPORT(rfilter);
    MI_PY_IMPORT(Stream);
    MI_PY_IMPORT(Bitmap);
    MI_PY_IMPORT(Formatter);
    MI_PY_IMPORT(FileResolver);
    MI_PY_IMPORT(Logger);
    MI_PY_IMPORT(MemoryMappedFile);
    MI_PY_IMPORT(DummyStream);
    MI_PY_IMPORT(FileStream);
    MI_PY_IMPORT(MemoryStream);
    MI_PY_IMPORT(ZStream);
    MI_PY_IMPORT(ProgressReporter);
    MI_PY_IMPORT(Thread);
    MI_PY_IMPORT(Timer);
    MI_PY_IMPORT(Properties);
    MI_PY_IMPORT(parser);
    MI_PY_IMPORT(misc);

    MI_PY_IMPORT(BSDFContext);
    MI_PY_IMPORT(EmitterExtras);
    MI_PY_IMPORT(RayFlags);
    MI_PY_IMPORT(MicrofacetType);
    MI_PY_IMPORT(PhaseFunctionExtras);
    MI_PY_IMPORT(Spiral);
    MI_PY_IMPORT(Sensor);
    MI_PY_IMPORT(FilmFlags);
    MI_PY_IMPORT(DiscontinuityFlags);

    /* Register a cleanup callback function to wait for pending tasks (this is
     * called before all Python variables are cleaned up */
    auto atexit = nb::module_::import_("atexit");
    atexit.attr("register")(nb::cpp_function([]() {
        {
            nb::gil_scoped_release g;
            Thread::wait_for_tasks();
        }

        // Release all loaded plugins
        PluginManager::instance()->release_all();

        StructConverter::static_shutdown();
        Profiler::static_shutdown();
        Bitmap::static_shutdown();
        Logger::static_shutdown();
        Thread::static_shutdown();
        Jit::static_shutdown();
    }));

    /* Make this a package, thus allowing statements such as:
     * `from mitsuba.test.util import function`
     * For that `__path__` needs to be populated. We do it by using the
     * `__file__` attribute of a Python file which is located in the same
     * directory as this module */
    nb::module_ os = nb::module_::import_("os");
    nb::module_ cfg = nb::module_::import_("mitsuba.config");
    nb::object cfg_path = os.attr("path").attr("realpath")(cfg.attr("__file__"));
    nb::object mi_dir = os.attr("path").attr("dirname")(cfg_path);
    nb::object mi_py_dir = os.attr("path").attr("join")(mi_dir, "python");
    nb::list paths{};
    paths.append(nb::str(mi_dir));
    paths.append(nb::str(mi_py_dir));
    m.attr("__path__") = paths;

    // Change module name back to correct value
    m.attr("__name__") = "mitsuba_ext";
}
