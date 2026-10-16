package gen

import (
	"fmt"
	"strings"

	"github.com/gsoultan/mimir/backend"
)

// The special functions are helpers a kernel defines only when its
// program calls one, in float arithmetic and the prelude's functions
// alone, so that both dialects compute them alike: neither Metal nor
// NVRTC without headers offers them all. They agree with the CPU
// backend's float64 functions to a few float32 ulps, or about 1e-6
// absolute where a result is near zero. Each is sp_ and the op's name.
const specialHelpers = `
DECL bool sp_isint(float x) {
	float a = x < 0.0f ? -x : x;
	return a >= 8388608.0f || x == float(int(x));
}
DECL float sp_log1p(float x) {
	float u = 1.0f + x;
	if (u == 1.0f) return x;
	if (u == fbits(0x7f800000u)) return u;
	return log(u) * (x / (u - 1.0f));
}
DECL float sp_expm1(float x) {
	float u = exp2(x * 1.44269504f);
	if (u == 1.0f) return x;
	float um = u - 1.0f;
	if (um == -1.0f) return -1.0f;
	if (u == fbits(0x7f800000u)) return u;
	return um * (x / log(u));
}
DECL float sp_erf(float x) {
	float a = x < 0.0f ? -x : x;
	if (a < 0.5f) {
		float s = x * x;
		float p = -1.32275132e-5f;
		p = p * s + 1.06837607e-4f;
		p = p * s - 7.57575758e-4f;
		p = p * s + 4.62962963e-3f;
		p = p * s - 2.38095238e-2f;
		p = p * s + 1.0e-1f;
		p = p * s - 3.33333333e-1f;
		p = p * s + 1.0f;
		return 1.12837917f * x * p;
	}
	float t = 1.0f / (1.0f + 0.5f * a);
	float q = 1.70872770e-1f;
	q = q * t - 8.22152230e-1f;
	q = q * t + 1.48851587f;
	q = q * t - 1.13520398f;
	q = q * t + 2.78868070e-1f;
	q = q * t - 1.86288060e-1f;
	q = q * t + 9.67841800e-2f;
	q = q * t + 3.74091960e-1f;
	q = q * t + 1.00002368f;
	float r = 1.0f - t * exp2((q * t - 1.26551223f - a * a) * 1.44269504f);
	return x < 0.0f ? -r : r;
}
DECL float sp_erfinv(float x) {
	float a = x < 0.0f ? -x : x;
	if (a >= 1.0f) return a == 1.0f ? x * fbits(0x7f800000u) : fbits(0x7fc00000u);
	float w = -log((1.0f - x) * (1.0f + x)), p;
	if (w < 5.0f) {
		w = w - 2.5f;
		p = 2.81022636e-08f;
		p = 3.43273939e-07f + p * w;
		p = -3.5233877e-06f + p * w;
		p = -4.39150654e-06f + p * w;
		p = 0.00021858087f + p * w;
		p = -0.00125372503f + p * w;
		p = -0.00417768164f + p * w;
		p = 0.246640727f + p * w;
		p = 1.50140941f + p * w;
	} else {
		w = sqrt(w) - 3.0f;
		p = -0.000200214257f;
		p = 0.000100950558f + p * w;
		p = 0.00134934322f + p * w;
		p = -0.00367342844f + p * w;
		p = 0.00573950773f + p * w;
		p = -0.0076224613f + p * w;
		p = 0.00943887047f + p * w;
		p = 1.00167406f + p * w;
		p = 2.83297682f + p * w;
	}
	return p * x;
}
DECL float sp_lgamma(float x) {
	float a = x < 0.0f ? -x : x;
	if (a == fbits(0x7f800000u)) return a;
	float refl = 0.0f, sign = 1.0f;
	if (x < 0.5f) {
		float s = sin_pi(x);
		s = s < 0.0f ? -s : s;
		if (s == 0.0f) return fbits(0x7f800000u);
		refl = log(3.14159265f / s);
		sign = -1.0f;
		x = 1.0f - x;
	}
	float p = 1.0f;
	while (x < 8.0f) {
		p *= x;
		x += 1.0f;
	}
	float z = 1.0f / x, z2 = z * z;
	float st = (x - 0.5f) * log(x) - x + 0.918938533f + z * (8.33333333e-2f - z2 * (2.77777778e-3f - z2 * 7.93650794e-4f));
	return refl + sign * (st - log(p));
}
DECL float sp_digamma(float x) {
	if (x == 0.0f) return (ubits(x) >> 31) != 0u ? fbits(0x7f800000u) : fbits(0xff800000u);
	if (x == fbits(0x7f800000u)) return x;
	float r = 0.0f;
	if (x < 0.0f) {
		if (sp_isint(x)) return fbits(0x7fc00000u);
		r = -3.14159265f * cos_pi(x) / sin_pi(x);
		x = 1.0f - x;
	}
	while (x < 6.0f) {
		r -= 1.0f / x;
		x += 1.0f;
	}
	float z = 1.0f / x, z2 = z * z;
	return r + log(x) - 0.5f * z - z2 * (8.33333333e-2f - z2 * (8.33333333e-3f - z2 * 3.96825397e-3f));
}
DECL float sp_trigamma(float x) {
	if (x <= 0.0f && sp_isint(x)) return fbits(0x7f800000u);
	if (x == fbits(0x7f800000u)) return 0.0f;
	float r = 0.0f, sign = 1.0f;
	if (x < 0.0f) {
		float s = sin_pi(x);
		r = 9.8696044f / (s * s);
		sign = -1.0f;
		x = 1.0f - x;
	}
	float t = 0.0f;
	while (x < 6.0f) {
		t += 1.0f / (x * x);
		x += 1.0f;
	}
	float z = 1.0f / x, z2 = z * z;
	t += z + 0.5f * z2 + z2 * z * (1.66666667e-1f - z2 * (3.33333333e-2f - z2 * 2.38095238e-2f));
	return r + sign * t;
}

`

// special returns the call of the special function op on x, and makes
// the kernel define the special functions.
func (g *gen) special(op backend.Op, x string) string {
	if !strings.Contains(g.helpers, "sp_isint") {
		g.helpers += strings.ReplaceAll(specialHelpers, "DECL ", g.d.decl+" ")
	}
	return fmt.Sprintf("sp_%s(%s)", op, x)
}
