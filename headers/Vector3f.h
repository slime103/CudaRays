#pragma once

#include <iostream>

class Vector3f
{
	public:
		float x;
		float y;
		float z;

		//Constructors
		Vector3f() {};

		Vector3f(float x, float y, float z);

		//addition and subtraction
		inline const Vector3f operator+(const Vector3f& v);
		inline const Vector3f operator-(const Vector3f& v);
		inline Vector3f& operator+=(const Vector3f& v);
		inline Vector3f& operator-=(const Vector3f& v);
		

		//scalar operations
		inline const Vector3f operator-();
		inline const Vector3f operator*(const float t);
		inline const Vector3f operator/(const float t);
		inline Vector3f& operator*=(const float t);
		inline Vector3f& operator/=(const float t);

		//length
		const float length();
		const float length_squared();

};

//printing
inline std::ostream& operator<<(std::ostream& out, const Vector3f& v);

//Additional operations

inline Vector3f operator*(const Vector3f& u, const Vector3f& v);

inline float dot(const Vector3f& u, const Vector3f& v);

inline Vector3f cross(const Vector3f& u, const Vector3f& v);

inline Vector3f normalize(Vector3f v);
