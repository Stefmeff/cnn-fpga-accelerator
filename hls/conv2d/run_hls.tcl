
set PROJ    "conv2d_proj"
set SOLN    "sol1"
set TOP     "conv2d_hls"
set PART    "xc7z020clg400-1"     ;# PYNQ-Z2 (matches config/hls.cfg)
set PERIOD  8                     ;# ns -> 125 MHz (matches config/hls.cfg)

set SRC     "conv2d.cpp"
set SRC_CFLAGS "-I."

# testbench sources (relative to this folder) + include paths for the
# project's reference conv2d() and the Tensor class.
set TB_SRCS [list \
    "tb_conv2d.cpp" \
    "../../src/kernels.cpp" \
    "../../utils/tensor.cpp" \
]
set TB_CFLAGS "-I. -I../../src -I../../utils"

# ---------------- parse -tclargs ----------------
set do_csim   0
set do_csynth 0
set do_cosim  0
set do_export 0

if {$argc == 0} {
    set do_csim 1
} else {
    foreach a $argv {
        switch -- $a {
            csim              { set do_csim   1 }
            csynth  - synth   { set do_csynth 1 }
            cosim             { set do_cosim  1 }
            export  - ip      { set do_export 1 }
            all               { set do_csim 1; set do_csynth 1; set do_cosim 1; set do_export 1 }
            clean {
                puts "\[run_hls\] removing project '$PROJ' ..."
                file delete -force $PROJ
                exit 0
            }
            default { puts "\[run_hls\] WARNING: unknown stage '$a' (ignored)" }
        }
    }
}

# dependency: cosim and export both need synthesized RTL
if {$do_cosim || $do_export} { set do_csynth 1 }

puts "==================== conv2d HLS flow ===================="
puts " top    : $TOP        src : $SRC"
puts " part   : $PART @ [expr 1000.0/$PERIOD] MHz"
puts " stages : csim=$do_csim  csynth=$do_csynth  cosim=$do_cosim  export=$do_export"
puts "========================================================"

# ---------------- project setup ----------------
open_project -reset $PROJ
set_top $TOP

add_files $SRC -cflags $SRC_CFLAGS
foreach tb $TB_SRCS {
    add_files -tb $tb -cflags $TB_CFLAGS
}

open_solution -reset $SOLN -flow_target vitis
set_part $PART
create_clock -period $PERIOD -name default

# ---------------- run requested stages ----------------
if {$do_csim}   { csim_design }
if {$do_csynth} { csynth_design }
if {$do_cosim}  { cosim_design }
if {$do_export} { export_design -format xo -output "conv2d.xo" }

puts "\[run_hls\] done."
exit
