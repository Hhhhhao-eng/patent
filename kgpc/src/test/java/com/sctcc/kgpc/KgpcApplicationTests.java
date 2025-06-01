package com.sctcc.kgpc;

import com.example.config.RestTemplateConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

@SpringBootTest(classes = {PatentController.class, RestTemplateConfig.class})
class PatentControllerTest {

    @Autowired
    private PatentController patentController;
    
    @MockBean
    private RestTemplate restTemplate;
    
    @BeforeEach
    void setup() {
        // 模拟成功的 Django 响应
        ResponseEntity<String> mockResponse = ResponseEntity.ok("[]");
        
        when(restTemplate.exchange(
                any(String.class), 
                eq(HttpMethod.GET), 
                any(), 
                eq(String.class)
        )).thenReturn(mockResponse);
    }
    
    @Test
    void patentRecommend_Success() {
        ResponseEntity<?> response = patentController.patentRecommend("CN100000");
        assertEquals(200, response.getStatusCode().value());
    }
    
    @Test
    void patentRecommend_FailsWhenDjangoUnavailable() {
        // 模拟失败响应
        when(restTemplate.exchange(
                any(String.class), 
                eq(HttpMethod.GET), 
                any(), 
                eq(String.class)
        )).thenThrow(new RuntimeException("Service unavailable"));
        
        ResponseEntity<?> response = patentController.patentRecommend("CN100000");
        assertEquals(500, response.getStatusCode().value());
    }
}
