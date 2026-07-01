#include "bmp2tensor.h"


static void write8(FILE * f, uint8_t val)
{
	if(1 != fwrite(&val,1,1,f))
		printf("ERROR failed to write Little Endian!\n");
}


static void write16(FILE * f, uint16_t val)
{
	write8(f,val & 0xFF);
	write8(f,val >> 8);
}


static void write32(FILE * f, uint32_t val)
{
	write16(f,val & 0xFFFF);
	write16(f,val >> 16);
}


uint8_t read8(FILE * f)
{
	uint8_t val;
	if( 1 != fread(&val,1,1,f)){
		printf("ERROR reading Little Endian!\n");
		return 0xFF;
	}
	return val;
}

uint16_t read16(FILE *f)
{
	uint16_t val = read8(f) | (read8(f) << 8);
	return val;
}

uint32_t read32(FILE * f)
{
	uint32_t val = read16(f) | (read16(f) << 16 );
	return val;
}





/* Write Bitmap info header 
 *
 * Can be converted to from imagemagick: convert imput.* -define bmp:format=bmp3 output.bmp
 *
 * 4: size header 40
 * 4: width (signed int)
 * 4: height (signed int)
 * 2: color planes
 * 2: bits per pixel (24 (3 channels))
 * 4: compression method (0: none) BI_RBG
 * 4: raw image size (can be 0 for no compression)
 * 4: horizontal pixel per meter ?
 * 4: vertical resolution pixel per meter
 * 4: number of colors in color palette (I guess 0)
 * 4: generally ignored set to 0
 */

constexpr uint32_t DEFAULT_PPM = 7000; // Roughly 4k 27 inch screen
struct bmp_info_s{
	uint32_t header_size = 40;
	uint32_t width;
	uint32_t height;
	uint16_t color_planes = 1;
	uint16_t bpp = 24;
	uint32_t compression_method = 0;
	uint32_t raw_image_size = 0;
	uint32_t res_hor = DEFAULT_PPM; // Default Value we don't intend to print anyways
	uint32_t res_ver = DEFAULT_PPM; // Default Value
	uint32_t n_palette = 0;
	uint32_t important_palette = 0;
};


static int readBMP3header(FILE *f, struct bmp_info_s * header)
{
	header->header_size = read32(f);
	if((header->header_size != 40) && (header->header_size != 124)){
		printf("WARNING header not BITMAPINFOHEADER (Size: %d)\n",
				header->header_size);
	}
	header->width = read32(f);
	header->height = read32(f);
	#ifdef BMP_DEBUG
		printf("Header size: %u \n",header->header_size);
		printf("Img: %ux%u \n",header->width,header->height);
	#endif
	header->color_planes = read16(f);
	if(header->color_planes != 1){
		printf("WARNING color planes != 1\n");
		return 1;
	}
	header->bpp = read16(f);
	if(header->bpp != 24){
		printf("ERROR wrong bpp: %d\n",header->bpp);
		return 1;
	}
	header->compression_method = read32(f);
	if(header->compression_method!= 0){
		printf("ERROR wrong compression method: %d\n",
				header->compression_method);
		return 1;
	}
	header->raw_image_size = read32(f);
	header->res_hor = read32(f);
	header->res_ver = read32(f);
#ifdef BMP_DEBUG
	printf("Resolution p/m H: %d, V: %d\n",header->res_hor,header->res_ver);
#endif 
	header->n_palette = read32(f);
	header->important_palette = read32(f);
	return 0;
}




/*
 * Can read both BMP3 and BMP4
 */
Tensor * readBMP(const char * infile)
{
	FILE * f = fopen(infile,"rb");
	uint16_t val = read16(f);
	if(val != 0x4d42){
		printf("ERROR wrong bmp val!\n");
		return NULL;
	}
	uint32_t size = read32(f);
#ifdef BMP_DEBUG
	printf("BMP size: %u \n",size);
#endif
	/* Discard 4 bytes */
	read32(f);
	uint32_t offset = read32(f);
	struct bmp_info_s bmp_info;
	if(readBMP3header(f,&bmp_info)){
		return NULL;
	}
	uint32_t width = bmp_info.width;
	uint32_t height = bmp_info.height;
	if(fseek(f,offset,SEEK_SET)){
		printf("ERROR fseek failed!\n");
		return NULL;
	}
	Tensor * t = new Tensor(3,height,width);
	if(t == NULL)
		return NULL;
	for(int i = height - 1; i >= 0; i--){
		for(int j = 0; j < width; j++){
			for(int k = 0; k < 3; k++){
				const uint8_t v = read8(f);
				(*t)[k][i][j] = (((float) v)/255.f - norm_mean[k])/norm_std[k];
			}
		}
		int tot = width * 3;
		while(tot  % 4){
			read8(f);
			tot++;
		}
	}
	fclose(f);
	return t;
}


