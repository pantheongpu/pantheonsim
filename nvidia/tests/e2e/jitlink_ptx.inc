// PTX modules shared by nvjitlink_paths.cpp and nvfatbin_paths.cpp: a kernel,
// k, that needs a library to link with, the library, and the variations the
// device linker's rules are checked with.

// The kernel. It calls add_base and reads base, which the library defines,
// and has a file-scope function and variable of its own, twice and scale --
// names the library uses for different ones. Each module must keep its own.
[[maybe_unused]] static const char* kMain = R"(
.version 7.1
.target sm_80
.address_size 64

.extern .func  (.param .b32 func_retval0) add_base
(
	.param .b32 add_base_param_0
)
;
.extern .global .align 4 .u32 base;
.global .align 4 .u32 scale = 3;

.func  (.param .b32 func_retval0) twice(
	.param .b32 twice_param_0
)
{
	.reg .b32 	%r<3>;
	ld.param.u32 	%r1, [twice_param_0];
	shl.b32 	%r2, %r1, 1;
	st.param.b32 	[func_retval0+0], %r2;
	ret;
}

.visible .entry k(
	.param .u64 k_param_0
)
{
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;
	ld.param.u64 	%rd1, [k_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %tid.x;
	{
	.param .b32 param0;
	st.param.b32 	[param0+0], %r1;
	.param .b32 retval0;
	call.uni (retval0), add_base, (param0);
	ld.param.b32 	%r2, [retval0+0];
	}
	mul.lo.s32 	%r3, %r2, 1000;
	{
	.param .b32 param0;
	st.param.b32 	[param0+0], %r1;
	.param .b32 retval0;
	call.uni (retval0), twice, (param0);
	ld.param.b32 	%r4, [retval0+0];
	}
	ld.global.u32 	%r5, [scale];
	mad.lo.s32 	%r6, %r5, %r4, %r3;
	ld.global.u32 	%r7, [base];
	add.s32 	%r8, %r6, %r7;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.u32 	[%rd4], %r8;
	ret;
}
)";

// add_base(x) = twice(x) * scale + base, with this module's twice (2x + 1)
// and scale (5): 10x + 12.
[[maybe_unused]] static const char* kLib = R"(
.version 7.1
.target sm_80
.address_size 64

.visible .global .align 4 .u32 base = 7;
.global .align 4 .u32 scale = 5;

.func  (.param .b32 func_retval0) twice(
	.param .b32 twice_param_0
)
{
	.reg .b32 	%r<4>;
	ld.param.u32 	%r1, [twice_param_0];
	shl.b32 	%r2, %r1, 1;
	or.b32  	%r3, %r2, 1;
	st.param.b32 	[func_retval0+0], %r3;
	ret;
}

.visible .func  (.param .b32 func_retval0) add_base(
	.param .b32 add_base_param_0
)
{
	.reg .b32 	%r<6>;
	ld.param.u32 	%r1, [add_base_param_0];
	{
	.param .b32 param0;
	st.param.b32 	[param0+0], %r1;
	.param .b32 retval0;
	call.uni (retval0), twice, (param0);
	ld.param.b32 	%r2, [retval0+0];
	}
	ld.global.u32 	%r3, [scale];
	ld.global.u32 	%r4, [base];
	mad.lo.s32 	%r5, %r3, %r2, %r4;
	st.param.b32 	[func_retval0+0], %r5;
	ret;
}
)";

// A second strong definition of add_base, and a weak one.
[[maybe_unused]] static const char* kDup = R"(
.version 7.1
.target sm_80
.address_size 64
.visible .func  (.param .b32 func_retval0) add_base(
	.param .b32 add_base_param_0
)
{
	.reg .b32 	%r<2>;
	mov.u32 	%r1, -1;
	st.param.b32 	[func_retval0+0], %r1;
	ret;
}
)";
[[maybe_unused]] static const char* kWeak = R"(
.version 7.1
.target sm_80
.address_size 64
.weak .func  (.param .b32 func_retval0) add_base(
	.param .b32 add_base_param_0
)
{
	.reg .b32 	%r<2>;
	mov.u32 	%r1, -2;
	st.param.b32 	[func_retval0+0], %r1;
	ret;
}
)";
// Code for a newer architecture than the link's.
[[maybe_unused]] static const char* kHopper = R"(
.version 7.8
.target sm_90
.address_size 64
.visible .func hopper_only()
{
	ret;
}
)";

[[maybe_unused]] static int expected(int t) { return (10 * t + 12) * 1000 + 2 * t * 3 + 7; }
