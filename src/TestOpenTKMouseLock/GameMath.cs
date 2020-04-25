using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace Vintagestory.API.MathTools
{
    /// <summary>
    /// A large set of useful game mathematics functions
    /// </summary>
    public class GameMath
    {
        public const float TWOPI = (float)Math.PI * 2;
        public const float PI = (float)Math.PI;
        public const float PIHALF = (float)Math.PI / 2;

        public const float DEG2RAD = (float)Math.PI / 180.0f;
        public const float RAD2DEG = 180.0f / (float)Math.PI;
        const uint murmurseed = 144;


        #region Standard Sin/Cos Methods
        public static float Sin(float value)
        {
            return (float)Math.Sin(value);
        }

        public static float Cos(float value)
        {
            return (float)Math.Cos(value);
        }

        public static float Acos(float value)
        {
            return (float)Math.Acos(value);
        }

        public static float Asin(float value)
        {
            return (float)Math.Asin(value);
        }


        public static float Tan(float value)
        {
            return (float)Math.Tan(value);
        }


        public static double Sin(double value)
        {
            return Math.Sin(value);
        }

        public static double Cos(double value)
        {
            return Math.Cos(value);
        }

        public static double Acos(double value)
        {
            return Math.Acos(value);
        }

        public static double Asin(double value)
        {
            return Math.Asin(value);
        }

        public static double Tan(double value)
        {
            return Math.Tan(value);
        }


        // http://www.java-gaming.org/index.php?PHPSESSID=cuplcuatuqd8i9imq70sf7kt41&topic=24191.0

        /// <summary>
        /// Faster Sin at the cost of lower accuracy
        /// </summary>
        /// <param name="rad"></param>
        /// <returns></returns>
        public static float FastSin(float rad)
        {
            return sinValues[(int)(rad * radToIndex) & SIN_MASK];
        }

        /// <summary>
        /// Faster Cos at the cost of lower accuracy
        /// </summary>
        /// <param name="rad"></param>
        /// <returns></returns>
        public static float FastCos(float rad)
        {
            return cosValues[(int)(rad * radToIndex) & SIN_MASK];
        }

        /// <summary>
        /// Faster Sin at the cost of lower accuracy
        /// </summary>
        /// <param name="deg"></param>
        /// <returns></returns>
        public static float FastSinDeg(float deg)
        {
            return sinValues[(int)(deg * degToIndex) & SIN_MASK];
        }

        /// <summary>
        /// Faster Cos at the cost of lower accuracy
        /// </summary>
        /// <param name="deg"></param>
        /// <returns></returns>
        public static float FastCosDeg(float deg)
        {
            return cosValues[(int)(deg * degToIndex) & SIN_MASK];
        }

        private static int SIN_BITS, SIN_MASK, SIN_COUNT;
        private static float radFull, radToIndex;
        private static float degFull, degToIndex;
        private static float[] sinValues, cosValues;

        static GameMath()
        {

            SIN_BITS = 12;
            SIN_MASK = ~(-1 << SIN_BITS);
            SIN_COUNT = SIN_MASK + 1;

            radFull = (float)(Math.PI * 2.0);
            degFull = (float)(360.0);
            radToIndex = SIN_COUNT / radFull;
            degToIndex = SIN_COUNT / degFull;

            sinValues = new float[SIN_COUNT];
            cosValues = new float[SIN_COUNT];

            for (int i = 0; i < SIN_COUNT; i++)
            {
                sinValues[i] = (float)Math.Sin((i + 0.5f) / SIN_COUNT * radFull);
                cosValues[i] = (float)Math.Cos((i + 0.5f) / SIN_COUNT * radFull);
            }

            // Four cardinal directions (credits: Nate)
            for (int i = 0; i < 360; i += 90)
            {
                sinValues[(int)(i * degToIndex) & SIN_MASK] = (float)Math.Sin(i * Math.PI / 180.0);
                cosValues[(int)(i * degToIndex) & SIN_MASK] = (float)Math.Cos(i * Math.PI / 180.0);
            }
        }
        #endregion

        #region Fast and Slow Sqrts

        // http://blog.wouldbetheologian.com/2011/11/fast-approximate-sqrt-method-in-c.html
        // http://www.lomont.org/Math/Papers/2003/InvSqrt.pdf
        /// <summary>
        /// Faster square root but at the cost of lower accuracy
        /// </summary>
        /// <param name="z"></param>
        /// <returns></returns>
        public static float FastSqrt(float z)
        {
            if (z == 0) return 0;
            FloatIntUnion u;
            u.tmp = 0;
            float xhalf = 0.5f * z;
            u.f = z;
            u.tmp = 0x5f375a86 - (u.tmp >> 1);
            u.f = u.f * (1.5f - xhalf * u.f * u.f);
            return u.f * z;
        }

        public static double FastSqrt(double z)
        {
            if (z == 0) return 0;
            DoubleLongUnion u;
            u.tmp = 0;
            double xhalf = 0.5 * z;
            u.f = z;
            u.tmp = 0x5f375a86 - (u.tmp >> 1);
            u.f = u.f * (1.5 - xhalf * u.f * u.f);
            return u.f * z;
        }


        public static float Sqrt(float value)
        {
            return (float)Math.Sqrt(value);
        }

        public static float Sqrt(double value)
        {
            return (float)Math.Sqrt(value);
        }

        #endregion

        #region Clamping

        /// <summary>
        /// Force val to be inside a certain range
        /// </summary>
        /// <param name="val"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static float Clamp(float val, float min, float max)
        {
            return val < min ? min : val > max ? max : val;
        }

        /// <summary>
        /// Force val to be inside a certain range
        /// </summary>
        /// <param name="val"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static int Clamp(int val, int min, int max)
        {
            return val < min ? min : val > max ? max : val;
        }

        /// <summary>
        /// Force val to be inside a certain range
        /// </summary>
        /// <param name="val"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static byte Clamp(byte val, byte min, byte max)
        {
            return val < min ? min : val > max ? max : val;
        }

        /// <summary>
        /// Force val to be inside a certain range
        /// </summary>
        /// <param name="val"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static double Clamp(double val, double min, double max)
        {
            return val < min ? min : val > max ? max : val;
        }

        /// <summary>
        /// Force val to be outside a certain range
        /// </summary>
        /// <param name="val"></param>
        /// <param name="atLeastNeg"></param>
        /// <param name="atLeastPos"></param>
        /// <returns></returns>
        public static int InverseClamp(int val, int atLeastNeg, int atLeastPos)
        {
            return val < atLeastPos ? atLeastPos : val > atLeastNeg ? atLeastNeg : val;
        }

        #endregion

        #region Modulo
        /// <summary>
        /// C#'s %-Operation is actually not modulo but remainder, so this is the actual modulo function that ensures positive numbers as return value
        /// </summary>
        /// <param name="k"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static int Mod(int k, int n)
        {
            return ((k %= n) < 0) ? k + n : k;
        }

        public static uint Mod(uint k, uint n)
        {
            return ((k %= n) < 0) ? k + n : k;
        }

        /// <summary>
        /// C#'s %-Operation is actually not modulo but remainder, so this is the actual modulo function that ensures positive numbers as return value
        /// </summary>
        /// <param name="k"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static float Mod(float k, float n)
        {
            return ((k %= n) < 0) ? k + n : k;
        }

        /// <summary>
        /// C#'s %-Operation is actually not modulo but remainder, so this is the actual modulo function that ensures positive numbers as return value
        /// </summary>
        /// <param name="k"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static double Mod(double k, double n)
        {
            return ((k %= n) < 0) ? k + n : k;
        }

        #endregion

        #region Interpolation

        /// <summary>
        /// Treats value as a statistical average. Example: 2.1 will turn into 2 90% of the times and into 3 10% of times.
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static int RoundRandom(Random rand, float value)
        {
            return (int)value + ((rand.NextDouble() < (value - (int)value)) ? 1 : 0);
        }

        /// <summary>
        /// Returns the shortest distance between 2 angles
        /// See also https://stackoverflow.com/a/14498790/1873041
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns></returns>
        public static float AngleDegDistance(float start, float end)
        {
            return ((((end - start) % 360) + 540) % 360) - 180;
        }

        /// <summary>
        /// Returns the shortest distance between 2 angles
        /// See also https://stackoverflow.com/a/14498790/1873041
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns></returns>
        public static float AngleRadDistance(float start, float end)
        {
            return ((((end - start) % TWOPI) + TWOPI + PI) % TWOPI) - PI;
        }

        /// <summary>
        /// Generates a gaussian blur kernel to be used when blurring something
        /// </summary>
        /// <param name="sigma"></param>
        /// <param name="size"></param>
        /// <returns></returns>
        public static double[,] GenGaussKernel(double sigma = 1, int size = 5)
        {
            double[,] kernel = new double[size, size];
            double mean = size / 2.0;
            double sum = 0.0; // For accumulating the kernel values

            for (int x = 0; x < size; ++x)
            {
                for (int y = 0; y < size; ++y)
                {
                    kernel[x, y] = Math.Exp(-0.5 * (Math.Pow((x - mean) / sigma, 2.0) + Math.Pow((y - mean) / sigma, 2.0)))
                                     / (2 * Math.PI * sigma * sigma);

                    // Accumulate the kernel values
                    sum += kernel[x, y];
                }
            }

            // Normalize the kernel
            for (int x = 0; x < size; ++x)
            {
                for (int y = 0; y < size; ++y)
                {
                    kernel[x, y] /= sum;
                }
            }

            return kernel;
        }

        /// <summary>
        /// Basic Bilinear Lerp
        /// </summary>
        /// <param name="topleft"></param>
        /// <param name="topright"></param>
        /// <param name="botleft"></param>
        /// <param name="botright"></param>
        /// <param name="x"></param>
        /// <param name="z"></param>
        /// <returns></returns>
        public static float BiLerp(float topleft, float topright, float botleft, float botright, float x, float z)
        {
            float top = topleft + (topright - topleft) * x;
            float bot = botleft + (botright - botleft) * x;
            return top + (bot - top) * z;
        }


        /// <summary>
        /// Basic Bilinear Lerp
        /// </summary>
        /// <param name="topleft"></param>
        /// <param name="topright"></param>
        /// <param name="botleft"></param>
        /// <param name="botright"></param>
        /// <param name="x"></param>
        /// <param name="z"></param>
        /// <returns></returns>
        public static double BiLerp(double topleft, double topright, double botleft, double botright, double x, double z)
        {
            double top = topleft + (topright - topleft) * x;
            double bot = botleft + (botright - botleft) * x;
            return top + (bot - top) * z;
        }


        /// <summary>
        /// Unlike the other implementation here, which uses the default "uniform"
        /// treatment of t, this computation is used to calculate the same values but
        /// introduces the ability to "parameterize" the t values used in the
        /// calculation. This is based on Figure 3 from
        /// http://www.cemyuksel.com/research/catmullrom_param/catmullrom.pdf
        /// </summary>
        /// <param name="t">the actual interpolation ratio from 0 to 1 representing the position between p1 and p2 to interpolate the value.</param>
        /// <param name="p">An array of double values of length 4, where interpolation occurs from p1 to p2.</param>
        /// <param name="time">An array of time measures of length 4, corresponding to each p value.</param>
        /// <returns></returns>
        public static double CPCatmullRomSplineLerp(double t, double[] p, double[] time)
        {
            double L01 = p[0] * (time[1] - t) / (time[1] - time[0]) + p[1] * (t - time[0]) / (time[1] - time[0]);
            double L12 = p[1] * (time[2] - t) / (time[2] - time[1]) + p[2] * (t - time[1]) / (time[2] - time[1]);
            double L23 = p[2] * (time[3] - t) / (time[3] - time[2]) + p[3] * (t - time[2]) / (time[3] - time[2]);
            double L012 = L01 * (time[2] - t) / (time[2] - time[0]) + L12 * (t - time[0]) / (time[2] - time[0]);
            double L123 = L12 * (time[3] - t) / (time[3] - time[1]) + L23 * (t - time[1]) / (time[3] - time[1]);
            double C12 = L012 * (time[2] - t) / (time[2] - time[1]) + L123 * (t - time[1]) / (time[2] - time[1]);
            return C12;
        }


        #endregion


        /// <summary>
        /// Returns a value between 0..1. Returns 0 if val is smaller than left or greater than right. For val == (left+right)/2 the return value is 1. Every other value is a linear interpolation based on the distance to the middle value. Ascii art representation:
        /// 
        ///1  |      /\
        ///   |     /  \
        ///0.5|    /    \
        ///   |   /      \
        ///   |  /        \
        ///0  __/__________\______
        ///  left          right
        /// </summary>
        /// <param name="val"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static float TriangleStep(int val, int left, int right)
        {
            float mid = (left + right) / 2;
            float range = (right - left) / 2;

            return Math.Max(0, 1 - Math.Abs(val - mid) / range);
        }

        /// <summary>
        /// Returns a value between 0..1. Returns 0 if val is smaller than left or greater than right. For val == (left+right)/2 the return value is 1. Every other value is a linear interpolation based on the distance to the middle value. Ascii art representation:
        /// 
        ///1  |      /\
        ///   |     /  \
        ///0.5|    /    \
        ///   |   /      \
        ///   |  /        \
        ///0  __/__________\______
        ///  left          right
        /// </summary>
        /// <param name="val"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static float TriangleStep(float val, float left, float right)
        {
            float mid = (left + right) / 2;
            float range = (right - left) / 2;

            return Math.Max(0, 1 - Math.Abs(val - mid) / range);
        }






        #region MinMax

        public static double Max(params double[] values)
        {
            double max = values[0];
            for (int i = 0; i < values.Length; i++)
            {
                max = Math.Max(max, values[i]);
            }
            return max;
        }

        public static float Max(params float[] values)
        {
            float max = values[0];
            for (int i = 0; i < values.Length; i++)
            {
                max = Math.Max(max, values[i]);
            }
            return max;
        }

        public static int Max(params int[] values)
        {
            int max = values[0];
            for (int i = 0; i < values.Length; i++)
            {
                max = Math.Max(max, values[i]);
            }
            return max;
        }

        public static int Min(params int[] values)
        {
            int min = values[0];
            for (int i = 0; i < values.Length; i++)
            {
                min = Math.Min(min, values[i]);
            }
            return min;
        }

        public static float Min(params float[] values)
        {
            float min = values[0];
            for (int i = 0; i < values.Length; i++)
            {
                min = Math.Min(min, values[i]);
            }
            return min;
        }

        #endregion

        #region Hashing

        /// <summary>
        /// See also https://msdn.microsoft.com/en-us/library/system.security.cryptography.md5%28v=vs.110%29.aspx
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static string Md5Hash(string input)
        {
            using (MD5 md5Hash = MD5.Create())
            {
                // Convert the input string to a byte array and compute the hash.
                byte[] data = md5Hash.ComputeHash(Encoding.UTF8.GetBytes(input));

                // Create a new Stringbuilder to collect the bytes
                // and create a string.
                StringBuilder sBuilder = new StringBuilder();

                // Loop through each byte of the hashed data 
                // and format each one as a hexadecimal string.
                for (int i = 0; i < data.Length; i++)
                {
                    sBuilder.Append(data[i].ToString("x2"));
                }

                // Return the hexadecimal string.
                return sBuilder.ToString();
            }
        }

        /// <summary>
        /// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static int oaatHashMany(int x, int count)
        {
            for (int i = 0; i < count; i++)
            {
                x += (x << 10);
                x ^= (x >> 6);
                x += (x << 3);
                x ^= (x >> 11);
                x += (x << 15);
            }
            return x;
        }

        /// <summary>
        /// Bob Jenkins' One-At-A-Time hashing algorithm
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static uint oaatHashUMany(uint x, int count)
        {
            for (int i = 0; i < count; i++)
            {
                x += (x << 10);
                x ^= (x >> 6);
                x += (x << 3);
                x ^= (x >> 11);
                x += (x << 15);
            }
            return x;
        }

        /// <summary>
        /// A really bad, but very fast hashing method.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static float PrettyBadHash(int x, int y)
        {
            return (float)Mod((x * 12.9898 + y * 78.233) * 43758.5453, 1);
        }

        /// <summary>
        /// A not so fast, but higher quality than oaatHash(). See also https://en.wikipedia.org/wiki/MurmurHash. Includes a modulo operation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <param name="mod"></param>
        /// <returns></returns>
        public static int MurmurHash3Mod(int x, int y, int z, int mod)
        {
            return GameMath.Mod(MurmurHash3(x, y, z), mod);
        }



        /// <summary>
        /// A not so fast, but higher quality than oaatHash(). See also https://en.wikipedia.org/wiki/MurmurHash
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <returns></returns>
        public static int MurmurHash3(int x, int y, int z)
        {
            const uint c1 = 0xcc9e2d51;
            const uint c2 = 0x1b873593;

            uint h1 = murmurseed;
            uint k1 = (uint)x;

            k1 = (uint)x;
            /* bitmagic hash */
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;

            h1 ^= k1;
            h1 = rotl32(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;

            k1 = (uint)y;
            /* bitmagic hash */
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;

            h1 ^= k1;
            h1 = rotl32(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;

            k1 = (uint)z;
            /* bitmagic hash */
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;

            h1 ^= k1;
            h1 = rotl32(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;

            // finalization, magic chants to wrap it all up
            h1 ^= 3;
            h1 = fmix(h1);

            unchecked //ignore overflow
            {
                return (int)h1;
            }
        }

        private static uint rotl32(uint x, byte r)
        {
            return (x << r) | (x >> (32 - r));
        }

        private static uint fmix(uint h)
        {
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            h *= 0xc2b2ae35;
            h ^= h >> 16;
            return h;
        }




        /// <summary>
        /// Quasirandom sequence by Martin Roberts (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/)
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public static double R2Sequence1D(int n)
        {
            double g = 1.6180339887498948482;
            double a1 = 1.0 / g;
            return (0.5 + a1 * n) % 1;
        }



        #endregion

        /// <summary>
        /// Assigns the value of x1 to x2 and vice versa
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        public static void FlipVal(ref int x1, ref int x2)
        {
            int tmp = x1;
            x2 = x1;
            x1 = tmp;
        }

        /// <summary>
        /// Assigns the value of x1 to x2 and vice versa
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        public static void FlipVal(ref double x1, ref double x2)
        {
            double tmp = x1;
            x2 = x1;
            x1 = tmp;
        }

        /// <summary>
        /// Performs a Fisher-Yates shuffle in linear time or O(n)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="rand"></param>
        /// <param name="array"></param>
        public static void Shuffle<T>(Random rand, T[] array)
        {
            int n = array.Length;        // The number of items left to shuffle (loop invariant).
            while (n > 1)
            {
                int k = rand.Next(n);  // 0 <= k < n.
                n--;                   // n is now the last pertinent index;
                T temp = array[n];     // swap array[n] with array[k] (does nothing if k == n).
                array[n] = array[k];
                array[k] = temp;
            }
        }


        /// <summary>
        /// Performs a Fisher-Yates shuffle in linear time or O(n)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="rand"></param>
        /// <param name="array"></param>
        public static void Shuffle<T>(Random rand, List<T> array)
        {
            int n = array.Count;        // The number of items left to shuffle (loop invariant).
            while (n > 1)
            {
                int k = rand.Next(n);  // 0 <= k < n.
                n--;                   // n is now the last pertinent index;
                T temp = array[n];     // swap array[n] with array[k] (does nothing if k == n).
                array[n] = array[k];
                array[k] = temp;
            }
        }





    }



    [StructLayout(LayoutKind.Explicit)]
    internal struct FloatIntUnion
    {
        [FieldOffset(0)]
        public float f;

        [FieldOffset(0)]
        public int tmp;
    }


    [StructLayout(LayoutKind.Explicit)]
    internal struct DoubleLongUnion
    {
        [FieldOffset(0)]
        public double f;

        [FieldOffset(0)]
        public long tmp;
    }
}
