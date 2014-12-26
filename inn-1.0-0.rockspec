package = "inn"
version = "1.0-0"

source = {
   url = "git://github.com/szagoruyko/imagine-nn",
   tag = "master"
}

description = {
   summary = "IMAGINE/LIGM torch nn repository",
   detailed = [[
Universite Paris-Est MLV Imagine laboratory nn routines
   ]],
   homepage = "https://github.com/szagoruyko/imagine-nn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn",
   "cunn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
