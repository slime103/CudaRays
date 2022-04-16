#include <Vector3f.h>
#include <math.h> 

Vector3f::Vector3f(float x, float y, float z)
{
		this->x = x;
		this->y = y;
		this->z = z;
}

inline const Vector3f Vector3f::operator+(const Vector3f& v)
{
	return Vector3f(x + v.x, y + v.y, z + v.z);
}

inline const Vector3f Vector3f::operator-(const Vector3f& v)
{
	return Vector3f(x - v.x, y - v.y, z - v.z);
}

inline Vector3f& Vector3f::operator+=(const Vector3f& v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

inline Vector3f& Vector3f::operator-=(const Vector3f& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

inline const Vector3f Vector3f::operator-()
{
	return Vector3f(-x, -y, -z);
};

inline const Vector3f Vector3f::operator*(const float t)
{
	return Vector3f(x * t, y * t, z * t);
}

inline const Vector3f Vector3f::operator/(const float t)
{
	return Vector3f(x, y, z) * (1 / t);
}

Vector3f& Vector3f::operator*=(const float t) 
{
	x *= t;
	y *= t;
	z *= t;
	return *this;
}

Vector3f& Vector3f::operator/=(const float t)
{
	return *this *= 1.f / t;
}

const float Vector3f::length() 
{
	return sqrt(length_squared());
}

const float Vector3f::length_squared()
{
	return x * x + y * y + z * z;
}

inline std::ostream& operator<<(std::ostream& out, const Vector3f& v) 
{
	return out << v.x << ' ' << v.y << ' ' << v.z;
}

inline Vector3f operator*(const Vector3f& u, const Vector3f& v) 
{
	return Vector3f(u.x * v.x, u.y * v.y, u.z * v.z);
}

inline float dot(const Vector3f& u, const Vector3f& v)
{
	return u.x * v.x
		+ u.y * v.y
		+ u.z * v.z;
}

inline Vector3f cross(const Vector3f& u, const Vector3f& v) {
	return Vector3f(u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x);
}

inline Vector3f normalize(Vector3f v) 
{
	return v / v.length();
}